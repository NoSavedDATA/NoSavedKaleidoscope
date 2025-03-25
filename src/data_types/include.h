#pragma once



enum Types {
  type_float = 0,
  type_tensor = 1,
  type_pinned_tensor = 2,
  type_object = 3,
  type_string = 4,
};

enum NameSolverTypes {
  type_self = 0,
  type_attr = 1,
  type_vec = 2,
  type_var = 3,
  type_object_name = 4,
  type_object_vec = 5
};
