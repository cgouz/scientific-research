package infrastructure

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config is the full application configuration. It is read from environment
// variables (optionally pre-filled from an env file); a few command-line flags
// override individual values.
type Config struct {
	AppEnv   string
	LogLevel string

	HTTPHost string
	HTTPPort int

	DB DBConfig

	// ResourcesDir points at a resources folder on disk. Empty means the
	// resources embedded in the binary are used.
	ResourcesDir   string
	MigrateOnStart bool
	SeedOnStart    bool
	// SeedCountries limits seeded organizations and sites to these ISO2
	// codes. Empty means all. Countries and categories are always seeded.
	SeedCountries []string

	Scheduler SchedulerConfig
	Crawler   CrawlerConfig
	LLM       LLMConfig
}

type DBConfig struct {
	Host         string
	Port         int
	Name         string
	User         string
	Password     string
	SSLMode      string
	MaxOpenConns int
}

type SchedulerConfig struct {
	Enabled      bool
	Interval     time.Duration
	MaxSites     int           // sites crawled at the same time
	FailureRetry time.Duration // delay before a failed site is retried (capped by its interval)
}

type CrawlerConfig struct {
	Workers      int
	Timeout      time.Duration
	MaxBodyBytes int64
	UserAgent    string
}

type LLMConfig struct {
	Runtime       string
	BaseURL       string
	Model         string
	APIKey        string
	Timeout       time.Duration
	Workers       int
	MaxRetries    int
	MaxInputChars int
	MaxTokens     int
}

func (c Config) HTTPAddr() string {
	return fmt.Sprintf("%s:%d", c.HTTPHost, c.HTTPPort)
}

// LoadConfig parses flags from args and reads the environment.
func LoadConfig(args []string) (Config, error) {
	return loadConfig(args, os.LookupEnv)
}

func loadConfig(args []string, lookup func(string) (string, bool)) (Config, error) {
	fs := flag.NewFlagSet("ggkip", flag.ContinueOnError)
	envFile := fs.String("env-file", "", "KEY=VALUE file with defaults; real environment variables win")
	runtime := fs.String("llm-runtime", "", "LLM runtime: vllm or llamacpp (overrides LLM_RUNTIME)")
	migrate := fs.String("migrate", "", "run migrations on start: true or false (overrides MIGRATE_ON_START)")
	seed := fs.String("seed", "", "load seed data on start: true or false (overrides SEED_ON_START)")
	if err := fs.Parse(args); err != nil {
		return Config{}, err
	}

	file := map[string]string{}
	if *envFile != "" {
		var err error
		if file, err = ReadEnvFile(*envFile); err != nil {
			return Config{}, err
		}
	}
	env := &envReader{lookup: func(key string) (string, bool) {
		if v, ok := lookup(key); ok {
			return v, true
		}
		v, ok := file[key]
		return v, ok
	}}

	cfg := Config{
		AppEnv:   env.str("APP_ENV", "local"),
		LogLevel: env.str("LOG_LEVEL", "info"),
		HTTPHost: env.str("HTTP_HOST", "0.0.0.0"),
		HTTPPort: env.int("HTTP_PORT", 8090),
		DB: DBConfig{
			Host:         env.str("DB_HOST", "localhost"),
			Port:         env.int("DB_PORT", 5432),
			Name:         env.str("DB_NAME", "ggkip"),
			User:         env.str("DB_USER", "ggkip"),
			Password:     env.str("DB_PASSWORD", ""),
			SSLMode:      env.str("DB_SSLMODE", "disable"),
			MaxOpenConns: env.int("DB_MAX_OPEN_CONNS", 20),
		},
		ResourcesDir:   env.str("RESOURCES_DIR", ""),
		MigrateOnStart: env.bool("MIGRATE_ON_START", true),
		SeedOnStart:    env.bool("SEED_ON_START", false),
		SeedCountries:  env.list("SEED_COUNTRIES"),
		Scheduler: SchedulerConfig{
			Enabled:      env.bool("SCHEDULER_ENABLED", true),
			Interval:     env.seconds("SCHEDULER_INTERVAL_SEC", 60),
			MaxSites:     env.int("SCHEDULER_MAX_SITES", 2),
			FailureRetry: env.seconds("SCHEDULER_FAILURE_RETRY_SEC", 1800),
		},
		Crawler: CrawlerConfig{
			Workers:      env.int("CRAWLER_WORKERS", 4),
			Timeout:      env.seconds("CRAWLER_TIMEOUT_SEC", 30),
			MaxBodyBytes: int64(env.int("CRAWLER_MAX_BODY_BYTES", 5<<20)),
			UserAgent:    env.str("CRAWLER_USER_AGENT", "GGKIPBot/0.1"),
		},
		LLM: LLMConfig{
			Runtime:       env.str("LLM_RUNTIME", "vllm"),
			BaseURL:       env.str("LLM_BASE_URL", "http://localhost:8000/v1"),
			Model:         env.str("LLM_MODEL", ""),
			APIKey:        env.str("LLM_API_KEY", ""),
			Timeout:       env.seconds("LLM_TIMEOUT_SEC", 180),
			Workers:       env.int("LLM_WORKERS", 4),
			MaxRetries:    env.int("LLM_MAX_RETRIES", 2),
			MaxInputChars: env.int("LLM_MAX_INPUT_CHARS", 24000),
			MaxTokens:     env.int("LLM_MAX_TOKENS", 4096),
		},
	}

	if *runtime != "" {
		cfg.LLM.Runtime = *runtime
	}
	cfg.LLM.Runtime = strings.ToLower(strings.TrimSpace(cfg.LLM.Runtime))
	if *migrate != "" {
		cfg.MigrateOnStart = env.parseBool("--migrate", *migrate)
	}
	if *seed != "" {
		cfg.SeedOnStart = env.parseBool("--seed", *seed)
	}

	if err := errors.Join(env.errs...); err != nil {
		return Config{}, err
	}
	return cfg, cfg.validate()
}

func (c Config) validate() error {
	var errs []error
	check := func(ok bool, msg string) {
		if !ok {
			errs = append(errs, errors.New(msg))
		}
	}
	check(c.HTTPPort > 0 && c.HTTPPort < 65536, "HTTP_PORT must be 1-65535")
	check(c.DB.MaxOpenConns > 0, "DB_MAX_OPEN_CONNS must be > 0")
	check(c.Scheduler.Interval > 0, "SCHEDULER_INTERVAL_SEC must be > 0")
	check(c.Scheduler.MaxSites > 0, "SCHEDULER_MAX_SITES must be > 0")
	check(c.Crawler.Workers > 0, "CRAWLER_WORKERS must be > 0")
	check(c.Crawler.Timeout > 0, "CRAWLER_TIMEOUT_SEC must be > 0")
	check(c.Crawler.MaxBodyBytes > 0, "CRAWLER_MAX_BODY_BYTES must be > 0")
	check(c.LLM.Workers > 0, "LLM_WORKERS must be > 0")
	check(c.LLM.MaxRetries >= 0, "LLM_MAX_RETRIES must be >= 0")
	check(c.LLM.MaxInputChars > 0, "LLM_MAX_INPUT_CHARS must be > 0")
	check(c.LLM.BaseURL != "", "LLM_BASE_URL is required")
	return errors.Join(errs...)
}

type envReader struct {
	lookup func(string) (string, bool)
	errs   []error
}

func (e *envReader) str(key, def string) string {
	if v, ok := e.lookup(key); ok && strings.TrimSpace(v) != "" {
		return strings.TrimSpace(v)
	}
	return def
}

// list reads a comma-separated value as upper-case items.
func (e *envReader) list(key string) []string {
	var out []string
	for _, v := range strings.Split(e.str(key, ""), ",") {
		if v = strings.ToUpper(strings.TrimSpace(v)); v != "" {
			out = append(out, v)
		}
	}
	return out
}

func (e *envReader) int(key string, def int) int {
	v := e.str(key, "")
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		e.errs = append(e.errs, fmt.Errorf("%s: %q is not an integer", key, v))
		return def
	}
	return n
}

func (e *envReader) seconds(key string, def int) time.Duration {
	return time.Duration(e.int(key, def)) * time.Second
}

func (e *envReader) bool(key string, def bool) bool {
	v := e.str(key, "")
	if v == "" {
		return def
	}
	return e.parseBool(key, v)
}

func (e *envReader) parseBool(name, v string) bool {
	b, err := strconv.ParseBool(v)
	if err != nil {
		e.errs = append(e.errs, fmt.Errorf("%s: %q is not a boolean", name, v))
	}
	return b
}
