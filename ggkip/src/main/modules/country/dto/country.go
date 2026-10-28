package dto

import "ggkip/src/main/modules/country/model"

type CountryResponse struct {
	ID           int64             `json:"id"`
	ISO2         string            `json:"iso2"`
	ISO3         string            `json:"iso3"`
	Name         string            `json:"name"`
	Soato        *string           `json:"soato"`
	Region       *string           `json:"region"`
	Subregion    *string           `json:"subregion"`
	IsActive     bool              `json:"is_active"`
	Translations map[string]string `json:"translations,omitempty"`
}

func FromModel(c model.Country) CountryResponse {
	return CountryResponse{
		ID: c.ID, ISO2: c.ISO2, ISO3: c.ISO3, Name: c.Name, Soato: c.Soato,
		Region: c.Region, Subregion: c.Subregion, IsActive: c.IsActive,
	}
}
