
#include "../codegen/string.h"
#include "../compiler_frontend/logging.h"
#include "include.h"

















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