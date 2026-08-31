# Node-20 continuation plan

1. Reproduce the result-parser outage path and prove that a completed training
   result with an emitted final metric is retained without a language-model
   parser response.
2. Restore persisted search-tree links and agent search state from the stopped
   run journal, clearing only stale in-flight locks.
3. Resume the existing Petfinder run at its recorded node count, execute the
   remaining nodes through the scheduler, and retain all prior artifacts.
4. Archive the completed run with scheduler traces and the required combined
   Gantt/metric figure, then push the record and source changes.
