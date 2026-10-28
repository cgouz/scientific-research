package model

import "time"

// Country is a row of countries. JSON tags match seed/countries.json.
type Country struct {
	ID        int64     `json:"id"`
	ISO2      string    `json:"iso2"`
	ISO3      string    `json:"iso3"`
	Name      string    `json:"name"`
	Soato     *string   `json:"soato"`
	Region    *string   `json:"region"`
	Subregion *string   `json:"subregion"`
	IsActive  bool      `json:"is_active"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// Translation is a row of country_translations (seed/countries_translations.json).
type Translation struct {
	ID        int64  `json:"id"`
	CountryID int64  `json:"country_id"`
	Locale    string `json:"locale"`
	Name      string `json:"name"`
}

type ListFilter struct {
	Region   *string
	IsActive *bool
	// Locale, when set, replaces Name with the translation for that locale if one exists.
	Locale *string
	Cursor int64
	Limit  int
}
