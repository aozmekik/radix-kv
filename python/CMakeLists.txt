# nanobind extension module `radix_kv._core`, built by scikit-build-core (`pip install .`) or by a
# plain CMake configure with -DRADIX_KV_BUILD_PYTHON=ON and nanobind importable from Python.
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_DIR
    COMMAND_ERROR_IS_FATAL ANY)
endif()
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core NB_STATIC src/radix_kv_module.cpp)
target_link_libraries(_core PRIVATE radix_kv::radix_kv)
# Our strict warnings apply to the bindings, not to nanobind's own headers.
set_target_properties(nanobind-static PROPERTIES SYSTEM ON)
radix_kv_set_warnings(_core)

# Type stubs generated from the compiled module, shipped next to it.
nanobind_add_stub(
  _core_stub
  MODULE _core
  OUTPUT _core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_core>
  DEPENDS _core)

install(TARGETS _core LIBRARY DESTINATION radix_kv)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_core.pyi DESTINATION radix_kv)
