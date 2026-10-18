add_executable(radix_kv_sim radix_kv_sim.cpp)
set_target_properties(radix_kv_sim PROPERTIES OUTPUT_NAME radix-kv-sim)
target_link_libraries(radix_kv_sim PRIVATE radix_kv::radix_kv)
radix_kv_set_warnings(radix_kv_sim)

if(RADIX_KV_INSTALL)
  include(GNUInstallDirs)
  install(TARGETS radix_kv_sim RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(RADIX_KV_BUILD_TESTS)
  add_test(NAME tool.radix_kv_sim.smoke
           COMMAND radix_kv_sim --requests 20 --blocks 256,1024 --max-running 8)
  set_tests_properties(tool.radix_kv_sim.smoke PROPERTIES PASS_REGULAR_EXPRESSION
                                                          "\\| rag \\| radix \\| 1024 \\|")
  add_test(NAME tool.radix_kv_sim.csv COMMAND radix_kv_sim --workload chat --strategy hash
                                              --requests 5 --format csv)
  set_tests_properties(tool.radix_kv_sim.csv PROPERTIES PASS_REGULAR_EXPRESSION
                                                        "chat,hash,2048,[0-9]+,")
  add_test(NAME tool.radix_kv_sim.help COMMAND radix_kv_sim --help)
  set_tests_properties(tool.radix_kv_sim.help PROPERTIES PASS_REGULAR_EXPRESSION "Usage:")
  add_test(NAME tool.radix_kv_sim.rejects_bad_flags COMMAND radix_kv_sim --blocks 0)
  set_tests_properties(tool.radix_kv_sim.rejects_bad_flags PROPERTIES WILL_FAIL TRUE)
  add_test(NAME tool.radix_kv_sim.rejects_empty_list_elements COMMAND radix_kv_sim --blocks 1024,)
  set_tests_properties(tool.radix_kv_sim.rejects_empty_list_elements PROPERTIES WILL_FAIL TRUE)
  # Every chat request carries a 1024-token system prompt, which does not fit 8 blocks.
  set(too_small --workload chat --strategy hash --requests 3 --blocks 8 --format csv)
  add_test(NAME tool.radix_kv_sim.reports_rejected_requests COMMAND radix_kv_sim ${too_small})
  set_tests_properties(tool.radix_kv_sim.reports_rejected_requests
                       PROPERTIES PASS_REGULAR_EXPRESSION "chat,hash,8,0,[1-9][0-9]*,")
  add_test(NAME tool.radix_kv_sim.fails_when_nothing_completes COMMAND radix_kv_sim ${too_small})
  set_tests_properties(tool.radix_kv_sim.fails_when_nothing_completes PROPERTIES WILL_FAIL TRUE)
endif()
