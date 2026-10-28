package model

import "time"

// Organization is a row of organizations (seed/organizations.json).
type Organization struct {
	ID                 int64     `json:"id"`
	CountryID          int64     `json:"country_id"`
	OrganizationTypeID int64     `json:"organization_type_id"`
	ParentID           *int64    `json:"parent_id"`
	Name               string    `json:"name"`
	OfficialName       *string   `json:"official_name"`
	Description        *string   `json:"description"`
	Status             string    `json:"status"`
	CreatedAt          time.Time `json:"created_at"`
	UpdatedAt          time.Time `json:"updated_at"`
}

const StatusActive = "active"

type ListFilter struct {
	CountryID          *int64
	OrganizationTypeID *int64
	ParentID           *int64
	Status             *string
	Cursor             int64
	Limit              int
}
