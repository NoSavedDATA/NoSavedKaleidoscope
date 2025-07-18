
#include "../codegen/string.h"
#include "../compiler_frontend/logging.h"
#include "include.h"



std::map<std::string, std::string> objectVecs;
std::map<std::string, int> objectVecsLastId;



extern "C" void InstantiateObject(char *obj_name, char *full_name)
{
  // std::cout << "\n\n\n\nInstantiateObject of: " << full_name << "\n\n\n";
  std::string _obj_name = obj_name;

  NamedObjects[full_name] = _obj_name + RandomString(13);
  // std::cout << "Saving " << NamedObjects[full_name]  << "\n\n";
}


extern "C" char *objHash(char *scope, char *obj_name)
{
  std::string _obj_name = obj_name;
  std::string ret = NamedObjects[scope+_obj_name];
  return str_to_char(ret);
}


extern "C" char *LoadObject(char *obj_name)
{
  // std::cout << "LOADING OBJECT " << obj_name << "\n";
  std::string ret = NamedObjects[obj_name];
  delete[] obj_name;
  // std::cout << "GOT OBJECT : " << ret << "\n";

  // std::cout << "Objects found: " << ".\n";
  // for (auto &pair : NamedObjects)
  // {
  //   std::cout << pair.first << " --> " << pair.second << ".\n";
  // }

  return str_to_char(ret);
}


extern "C" float InitObjectVecWithNull(char *name, float vec_size) 
{
  std::cout << "InitObjectVecWithNull of " << name << " with vec_size " << vec_size << "\n\n\n\n";

  for (int i=0; i<vec_size; i++)
  {
    std::string indexed_name = name + std::to_string(i);
    objectVecs[indexed_name] = "nullptr";
  }
  
  delete[] name; //TODO: Break?
  return 0;
}


extern "C" float is_null(char *name)
{
  //std::cout << "\n\nIS NULL OF: " << name << "\n\n\n";

  if (objectVecs[name]=="nullptr")
    return 1;
  return 0;
}




extern "C" void objAttr_var_from_var(char *LName, char *RName)
{
  //std::cout << "objAttr_var_from_var of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << NamedObjects[RName] << "\n";
  //std::cout << "Replacing: " << NamedObjects[LName] << "\n";

  NamedObjects[LName] = NamedObjects[RName];
  
  
}

extern "C" void objAttr_var_from_vec(char *LName, char *RName)
{
  //std::cout << "objAttr_var_from_vec of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << objectVecs[RName] << "\n";
  //std::cout << "Replacing: " << NamedObjects[LName] << "\n";

  NamedObjects[LName] = objectVecs[RName];

  
}

extern "C" void objAttr_vec_from_var(char *LName, char *RName)
{
  //std::cout << "objAttr_vec_from_var of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << NamedObjects[RName] << "\n";
  //std::cout << "Replacing: " << objectVecs[LName] << "\n";

  objectVecs[LName] = NamedObjects[RName];

  
}


extern "C" void objAttr_vec_from_vec(char *LName, char *RName)
{
  //std::cout << "objAttr_vec_from_vec of " << LName << " from " << RName << "\n";

  //std::cout << "Loading: " << objectVecs[RName] << "\n";
  //std::cout << "Replacing: " << objectVecs[LName] << "\n";

  objectVecs[LName] = objectVecs[RName];

  
}

extern "C" float append(char *self, char *obj_name)
{
  //char* copied = (char*)malloc(strlen(in_str) + 1);
  //strcpy(copied, in_str);

  std::cout << "\n\nAPPEND OF " << obj_name << " into: " << self << "\n";
  
  std::string obj_name_str = obj_name;


  


  int obj_vec_last_id = 0;
  if (objectVecsLastId.count(self)>0)
  {
    obj_vec_last_id = objectVecsLastId[self];
    obj_vec_last_id+=1;
  }
  objectVecsLastId[self] = obj_vec_last_id;

  std::string indexed_self = self + std::to_string(obj_vec_last_id);
  objectVecs[indexed_self] = NamedObjects[obj_name];

  
  

  return 0;
}

extern "C" char *LoadObjectScopeName(char *self)
{
  if (objectVecs.count(self)==0)
  {
    std::string _self = self;
    std::string _error = "Object "+_self+" does not exist";
    LogErrorS(_error);
    return "";
  }

  /*
  for (auto &pair : objectVecs)
  {
    std::cout <<  pair.first << ": " << pair.second << "\n";
  }
  */
  std::string ret = objectVecs[self];
  if(ret.length()==0)
  {
    for (auto &pair : objectVecs)
      std::cout <<  pair.first << ": " << pair.second << "\n";

    std::string _self = self;
    std::string _error = "Loaded object "+_self+" has zero length.";
    LogErrorS(_error);
  }


  //std::cout << "LoadObjectScopeName is: " << ret << ", from self: " << self << "\n";

  delete[] self;

  return str_to_char(ret);
}


extern "C" void *offset_object_ptr(void *object_ptr, int offset) {
  // std::cout << "offset object ptr of " << object_ptr << " on offset " << offset << ".\n";
  // std::cout << "offset_object_ptr" << ((char*)object_ptr + offset) << ".\n\n";
  
  return ((char*)object_ptr + offset);
}


extern "C" void object_Attr_float(void *object_ptr, float value) {
  // std::cout << "object_Attr_float" << ".\n";
  *(float *)((char*)object_ptr) = value;
}
extern "C" void object_Attr_int(void *object_ptr, int value) {
  // std::cout << "object_Attr_int" << ".\n";
  *(int *)((char*)object_ptr) = value;
}


extern "C" float object_Load_float(void *object_ptr) {
  float value = *(float*)((char*)object_ptr);
  return value;
}
extern "C" int object_Load_int(void *object_ptr) {
  int value = *(int*)((char*)object_ptr);
  return value;
}


extern "C" void *object_Load_slot(void *object_ptr) {
  // Read a void* attribute stored at object_ptr
  void **slot = (void **)((char *)object_ptr);
  // std::cout << "Loading slot " << *slot << " from " << object_ptr << ".\n";

  return *slot;
}




extern "C" void tie_object_to_object(void *object_ptr, void *object_attribute) {
  // Write the pointer to object_ptr + offset
  void **slot = (void **)((char *)object_ptr);
  *slot = object_attribute;
  // std::cout << "Attributing " << object_attribute << " to " << object_ptr << " on offset " << offset << ".\n";
}








extern "C" void object_Attr_on_Offset_float(Scope_Struct *scope_struct, float value, int offset) {
  *(float *)((char*)scope_struct->object_ptr + offset) = value;
}
extern "C" void object_Attr_on_Offset_int(Scope_Struct *scope_struct, int value, int offset) {
  *(int *)((char*)scope_struct->object_ptr + offset) = value;
}
extern "C" void object_Attr_on_Offset(Scope_Struct *scope_struct, void *value, int offset) {
  // std::cout << "STORING VOID " << value << " ON OFFSET " << offset << " on object " << scope_struct->object_ptr  << ".\n";
  *(void**)((char*)scope_struct->object_ptr + offset) = value;
}

extern "C" float object_Load_on_Offset_float(Scope_Struct *scope_struct, int offset) {
  float value = *(float*)((char*)scope_struct->object_ptr + offset);
  return value;
}
extern "C" int object_Load_on_Offset_int(Scope_Struct *scope_struct, int offset) {
  int value = *(int*)((char*)scope_struct->object_ptr + offset);
  return value;
}

extern "C" void *object_Load_on_Offset(Scope_Struct *scope_struct, int offset) {
  void **slot = (void **)((char *)scope_struct->object_ptr + offset);
  // std::cout << "Loading " << slot << " from " << scope_struct->object_ptr << " on offset " << offset << ".\n"; 
  
  return *slot;
}


extern "C" void *object_ptr_Load_on_Offset(void *object_ptr, int offset) {
  // Read a void* stored at object_ptr + offset
  void **slot = (void **)((char *)object_ptr + offset);
  // std::cout << "Loading " << slot << " from " << object_ptr << " on offset " << offset << ".\n";

  return *slot;
}

extern "C" void object_ptr_Attribute_object(void *object_ptr, int offset, void *object_attribute) {
  // Write the pointer to object_ptr + offset
  void **slot = (void **)((char *)object_ptr + offset);
  *slot = object_attribute;
  // std::cout << "Attributing " << object_attribute << " to " << object_ptr << " on offset " << offset << ".\n";
}