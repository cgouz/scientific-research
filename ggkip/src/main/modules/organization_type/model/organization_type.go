package model

import "time"

// OrganizationType is a row of organization_types (seed/organization_types.json).
type OrganizationType struct {
	ID        int64     `json:"id"`
	Code      string    `json:"code"`
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
}
