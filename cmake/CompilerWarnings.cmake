# Strict warning flags for the project's own targets (never applied to dependencies).
function(radix_kv_set_warnings target)
  if(MSVC)
    set(flags /W4 /permissive-)
    if(RADIX_KV_WARNINGS_AS_ERRORS)
      list(APPEND flags /WX)
    endif()
  else()
    set(flags
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Woverloaded-virtual
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
        -Wmissing-declarations)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      list(APPEND flags -Wduplicated-cond -Wduplicated-branches -Wlogical-op)
    endif()
    if(RADIX_KV_WARNINGS_AS_ERRORS)
      list(APPEND flags -Werror)
    endif()
  endif()
  target_compile_options(${target} PRIVATE ${flags})
endfunction()
