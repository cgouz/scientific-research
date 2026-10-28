// Command ggkip crawls official government websites, normalizes their text
// with an LLM into a fixed taxonomy and stores the records in PostgreSQL.
//
// main only wires dependencies and starts the application; the logic lives
// in the modules.
package main

import (
	"context"
	"database/sql"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"ggkip/src/main/app"
	"ggkip/src/main/infrastructure"
	categoryhandler "ggkip/src/main/modules/category/handler"
	categoryrepo "ggkip/src/main/modules/category/repo"
	categoryservice "ggkip/src/main/modules/category/service"
	countryhandler "ggkip/src/main/modules/country/handler"
	countryrepo "ggkip/src/main/modules/country/repo"
	countryservice "ggkip/src/main/modules/country/service"
	crawlhandler "ggkip/src/main/modules/crawl/handler"
	crawlrepo "ggkip/src/main/modules/crawl/repo"
	crawlservice "ggkip/src/main/modules/crawl/service"
	crawlermodel "ggkip/src/main/modules/crawler/model"
	crawlerservice "ggkip/src/main/modules/crawler/service"
	extractorservice "ggkip/src/main/modules/extractor/service"
	llmservice "ggkip/src/main/modules/llm/service"
	organizationhandler "ggkip/src/main/modules/organization/handler"
	organizationrepo "ggkip/src/main/modules/organization/repo"
	organizationservice "ggkip/src/main/modules/organization/service"
	orgtypehandler "ggkip/src/main/modules/organization_type/handler"
	orgtyperepo "ggkip/src/main/modules/organization_type/repo"
	orgtypeservice "ggkip/src/main/modules/organization_type/service"
	pagehandler "ggkip/src/main/modules/page/handler"
	pagerepo "ggkip/src/main/modules/page/repo"
	pageservice "ggkip/src/main/modules/page/service"
	recordhandler "ggkip/src/main/modules/record/handler"
	recordrepo "ggkip/src/main/modules/record/repo"
	recordservice "ggkip/src/main/modules/record/service"
	sitehandler "ggkip/src/main/modules/site/handler"
	siterepo "ggkip/src/main/modules/site/repo"
	siteservice "ggkip/src/main/modules/site/service"
	"ggkip/src/resources"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "ggkip:", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := infrastructure.LoadConfig(os.Args[1:])
	if err != nil {
		return err
	}
	log := infrastructure.NewLogger(cfg.AppEnv, cfg.LogLevel)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	resources, err := resourcesFS(cfg.ResourcesDir)
	if err != nil {
		return err
	}
	settings, err := app.LoadSettings(resources)
	if err != nil {
		return err
	}
	llmClient, err := llmservice.NewClient(cfg.LLM.Runtime, llmservice.ClientConfig{
		BaseURL: cfg.LLM.BaseURL, Model: cfg.LLM.Model, APIKey: cfg.LLM.APIKey,
		Timeout: cfg.LLM.Timeout, MaxTokens: cfg.LLM.MaxTokens,
	})
	if err != nil {
		return err
	}

	db, err := infrastructure.OpenPostgres(ctx, cfg.DB, log)
	if err != nil {
		return err
	}
	defer db.Close()

	if err := prepareDatabase(ctx, cfg, db, resources, log); err != nil {
		return err
	}

	// Repositories and services.
	countries := countryservice.New(countryrepo.New(db))
	orgTypes := orgtypeservice.New(orgtyperepo.New(db))
	organizations := organizationservice.New(organizationrepo.New(db))
	categories := categoryservice.New(categoryrepo.New(db))
	sites := siteservice.New(siterepo.New(db), cfg.Scheduler.FailureRetry)
	pages := pageservice.New(pagerepo.New(db))
	records := recordservice.New(recordrepo.New(db))

	// Crawl pipeline.
	crawler := crawlerservice.New(settings.Crawler, crawlermodel.Options{
		Workers: cfg.Crawler.Workers, Timeout: cfg.Crawler.Timeout,
		MaxBodyBytes: cfg.Crawler.MaxBodyBytes, UserAgent: cfg.Crawler.UserAgent,
	}, log)
	normalizer := llmservice.NewNormalizer(llmClient, cfg.LLM.Workers, cfg.LLM.MaxRetries, cfg.LLM.MaxInputChars, log)
	runner := crawlservice.NewRunner(ctx, crawlservice.RunnerDeps{
		DB: db, Sites: sites, Countries: countries, Organizations: organizations, Categories: categories,
		Crawler: crawler, Normalizer: normalizer, TaxonomySettings: settings.Taxonomy,
		Filter:     extractorservice.Filter{MinChars: settings.Crawler.MinTextChars, MinWords: settings.Crawler.MinTextWords},
		LLMWorkers: cfg.LLM.Workers, MaxSites: cfg.Scheduler.MaxSites, Log: log,
	})
	crawls := crawlservice.New(crawlrepo.New(db), sites, runner)

	// HTTP API.
	mux := http.NewServeMux()
	app.RegisterHealth(mux, db, llmClient.Runtime())
	countryhandler.New(countries, log).Register(mux)
	orgtypehandler.New(orgTypes, log).Register(mux)
	organizationhandler.New(organizations, log).Register(mux)
	categoryhandler.New(categories, log).Register(mux)
	sitehandler.New(sites, log).Register(mux)
	pagehandler.New(pages, log).Register(mux)
	crawlhandler.New(crawls, log).Register(mux)
	recordhandler.New(records, log).Register(mux)

	log.Info("ggkip starting", "env", cfg.AppEnv, "llm_runtime", llmClient.Runtime(), "llm_model", llmClient.Model(),
		"scheduler", cfg.Scheduler.Enabled)

	var wg sync.WaitGroup
	if cfg.Scheduler.Enabled {
		scheduler := app.NewScheduler(sites, runner, cfg.Scheduler.Interval, cfg.Scheduler.MaxSites, log)
		wg.Go(func() { scheduler.Run(ctx) })
	}
	serverErr := app.RunServer(ctx, cfg.HTTPAddr(), app.WithLogging(mux, log), log)
	stop() // the server can also stop on its own (e.g. port in use); stop everything else too

	wg.Wait()
	runner.Wait()
	log.Info("ggkip stopped")
	return serverErr
}

func prepareDatabase(ctx context.Context, cfg infrastructure.Config, db *sql.DB, resources fs.FS, log *slog.Logger) error {
	if cfg.MigrateOnStart {
		migrations, err := fs.Sub(resources, "migrations")
		if err != nil {
			return err
		}
		n, err := infrastructure.Migrate(ctx, db, migrations, log)
		if err != nil {
			return err
		}
		log.Info("migrations done", "applied", n)
	}
	if cfg.SeedOnStart {
		data, err := app.LoadSeed(resources)
		if err != nil {
			return err
		}
		if err := app.ApplySeed(ctx, db, data, log); err != nil {
			return err
		}
	}
	if n, err := crawlrepo.New(db).FailInterrupted(ctx); err != nil {
		return err
	} else if n > 0 {
		log.Warn("closed crawl runs interrupted by a previous stop", "count", n)
	}
	return nil
}

// resourcesFS returns the resources folder on disk when dir is set,
// otherwise the copy embedded in the binary.
func resourcesFS(dir string) (fs.FS, error) {
	if dir != "" {
		if _, err := os.Stat(dir); err != nil {
			return nil, fmt.Errorf("RESOURCES_DIR: %w", err)
		}
		return os.DirFS(dir), nil
	}
	return resources.FS, nil
}
