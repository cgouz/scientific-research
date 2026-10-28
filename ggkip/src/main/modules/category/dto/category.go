package dto

import "ggkip/src/main/modules/category/model"

type CategoryResponse struct {
	ID            int64              `json:"id"`
	ParentID      *int64             `json:"parent_id"`
	Code          string             `json:"code"`
	Name          string             `json:"name"`
	Description   *string            `json:"description"`
	IsActive      bool               `json:"is_active"`
	Subcategories []CategoryResponse `json:"subcategories,omitempty"`
}

func FromModel(c model.Category) CategoryResponse {
	return CategoryResponse{
		ID: c.ID, ParentID: c.ParentID, Code: c.Code, Name: c.Name, Description: c.Description, IsActive: c.IsActive,
	}
}

// Tree nests subcategories under their parent categories, keeping input order.
func Tree(items []model.Category) []CategoryResponse {
	var roots []CategoryResponse
	index := map[int64]int{}
	for _, c := range items {
		if c.ParentID == nil {
			index[c.ID] = len(roots)
			roots = append(roots, FromModel(c))
		}
	}
	for _, c := range items {
		if c.ParentID != nil {
			if i, ok := index[*c.ParentID]; ok {
				roots[i].Subcategories = append(roots[i].Subcategories, FromModel(c))
			}
		}
	}
	return roots
}
