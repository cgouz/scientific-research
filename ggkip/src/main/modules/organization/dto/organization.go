package dto

import "ggkip/src/main/modules/organization/model"

type OrganizationResponse struct {
	ID                 int64   `json:"id"`
	CountryID          int64   `json:"country_id"`
	OrganizationTypeID int64   `json:"organization_type_id"`
	ParentID           *int64  `json:"parent_id"`
	Name               string  `json:"name"`
	OfficialName       *string `json:"official_name"`
	Description        *string `json:"description"`
	Status             string  `json:"status"`
}

func FromModel(o model.Organization) OrganizationResponse {
	return OrganizationResponse{
		ID: o.ID, CountryID: o.CountryID, OrganizationTypeID: o.OrganizationTypeID, ParentID: o.ParentID,
		Name: o.Name, OfficialName: o.OfficialName, Description: o.Description, Status: o.Status,
	}
}
