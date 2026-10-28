package infrastructure

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"
	"time"
)

func TestSHA256KnownVectors(t *testing.T) {
	for in, want := range map[string]string{
		"":    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		"abc": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
	} {
		if got := hex.EncodeToString(SHA256(in)); got != want {
			t.Errorf("SHA256(%q) = %s, want %s", in, got, want)
		}
	}
}

func lookup(env map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) { v, ok := env[k]; return v, ok }
}

func TestConfigDefaults(t *testing.T) {
	cfg, err := loadConfig(nil, lookup(nil))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.Runtime != "vllm" || cfg.HTTPPort != 8090 || cfg.Scheduler.Interval != 60*time.Second {
		t.Fatalf("unexpected defaults: %+v", cfg)
	}
}

func TestConfigRuntimeFlagOverridesEnv(t *testing.T) {
	env := map[string]string{"LLM_RUNTIME": "vllm"}
	cfg, err := loadConfig([]string{"--llm-runtime=llamacpp"}, lookup(env))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.Runtime != "llamacpp" {
		t.Fatalf("runtime = %q, want llamacpp", cfg.LLM.Runtime)
	}

	cfg, err = loadConfig(nil, lookup(map[string]string{"LLM_RUNTIME": " LlamaCpp "}))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.LLM.Runtime != "llamacpp" {
		t.Fatalf("runtime from env = %q, want llamacpp", cfg.LLM.Runtime)
	}
}

func TestConfigEnvFileAndPrecedence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.env")
	content := "# comment\nexport HTTP_PORT=9000\nDB_NAME=\"from_file\"\nLLM_WORKERS=3\n"
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := loadConfig([]string{"--env-file", path}, lookup(map[string]string{"LLM_WORKERS": "8"}))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.HTTPPort != 9000 || cfg.DB.Name != "from_file" {
		t.Fatalf("env file not applied: port=%d db=%q", cfg.HTTPPort, cfg.DB.Name)
	}
	if cfg.LLM.Workers != 8 {
		t.Fatalf("real environment must win over the env file: workers=%d", cfg.LLM.Workers)
	}
}

func TestConfigCrawlLimits(t *testing.T) {
	cfg, err := loadConfig(nil, lookup(map[string]string{"SCHEDULER_MAX_SITES": "3"}))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Crawl.MaxSites != 3 || cfg.Crawl.MaxSitesPerCountry != 4 {
		t.Fatalf("CRAWL_MAX_SITES must default to SCHEDULER_MAX_SITES (old setups): %+v", cfg.Crawl)
	}
	cfg, err = loadConfig(nil, lookup(map[string]string{"CRAWL_MAX_SITES": "16", "CRAWL_MAX_SITES_PER_COUNTRY": "8"}))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Crawl.MaxSites != 16 || cfg.Crawl.MaxSitesPerCountry != 8 {
		t.Fatalf("crawl limits not read: %+v", cfg.Crawl)
	}
	if _, err := loadConfig(nil, lookup(map[string]string{"CRAWL_MAX_SITES_PER_COUNTRY": "0"})); err == nil {
		t.Fatal("expected error for CRAWL_MAX_SITES_PER_COUNTRY=0")
	}
}

func TestConfigRejectsBadValues(t *testing.T) {
	if _, err := loadConfig(nil, lookup(map[string]string{"HTTP_PORT": "abc"})); err == nil {
		t.Fatal("expected error for non-integer HTTP_PORT")
	}
	if _, err := loadConfig(nil, lookup(map[string]string{"CRAWLER_WORKERS": "0"})); err == nil {
		t.Fatal("expected error for CRAWLER_WORKERS=0")
	}
}

func TestListMigrations(t *testing.T) {
	fsys := fstest.MapFS{
		"002_b.sql": {Data: []byte("SELECT 2")},
		"001_a.sql": {Data: []byte("SELECT 1")},
		"010_c.sql": {Data: []byte("SELECT 10")},
		"README.md": {Data: []byte("ignored")},
	}
	got, err := listMigrations(fsys)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0].Filename != "001_a.sql" || got[2].Version != 10 {
		t.Fatalf("unexpected order: %+v", got)
	}

	if _, err := listMigrations(fstest.MapFS{"1_a.sql": {}, "01_b.sql": {}}); err == nil {
		t.Fatal("expected error for duplicate versions")
	}
	if _, err := listMigrations(fstest.MapFS{"create.sql": {}}); err == nil {
		t.Fatal("expected error for a file without version prefix")
	}
}

func TestRealMigrationsAreOrdered(t *testing.T) {
	got, err := listMigrations(os.DirFS("../../resources/migrations"))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || got[0].Filename != "001_create_countries.sql" {
		t.Fatalf("unexpected migrations: %+v", got)
	}
}
