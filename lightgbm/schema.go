package lightgbm

type featureInfo struct {
	MinValue float64  `json:"min_value"`
	MaxValue float64  `json:"max_value"`
	Values   []string `json:"values"` //TODO has to check this type of this
}

type treeNode struct {
	SplitIndex     int       `json:"split_index"`
	SplitFeature   int       `json:"split_feature"`
	SplitGain      float64   `json:"split_gain"`
	Threshold      float64   `json:"threshold"`
	DecisionType   string    `json:"decision_type"`
	DefaultLeft    bool      `json:"default_left"`
	MissingType    string    `json:"missing_type"`
	InternalValue  float64   `json:"internal_value"`
	InternalWeight float64   `json:"internal_weight"`
	InternalCount  int       `json:"internal_count"`
	LeftChild      *treeNode `json:"left_child"`
	RightChild     *treeNode `json:"right_child"`
}

type tree struct {
	TreeIndex     int      `json:"tree_index"`
	NumLeaves     int      `json:"num_leaves"`
	NumCat        int      `json:"num_cat"`
	Shrinkage     float64  `json:"shrinkage"`
	TreeStructure treeNode `json:"tree_structure"`
}

type gbdtModel struct {
	Name                string                 `json:"name"`
	Version             string                 `json:"version"`
	NumClass            int                    `json:"num_class"`
	NumTreePerIteration int                    `json:"num_tree_per_iteration"`
	LabelIndex          int                    `json:"label_index"`
	MaxFeatureIdx       int                    `json:"max_feature_idx"`
	Objective           string                 `json:"objective"`
	AverageOutput       bool                   `json:"average_output"`
	FeatureNames        []string               `json:"feature_names"`
	MonotoneConstraints []string               `json:"monotone_constraints"` // TODO: check the type
	FeatureInfos        map[string]featureInfo `json:"feature_infos"`
	FeatureImportance   map[string]int         `json:"feature_importances"`
	PandasCategorical   []string               `json:"pandas_categorical"` // TODO: check the type
	TreeInfo            []tree                 `json:"tree_info"`
}
