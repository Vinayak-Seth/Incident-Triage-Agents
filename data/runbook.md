# Incident runbook

Each entry: symptoms, likely cause, how to confirm, fix. The investigator agent
searches this file with the `search_runbook` tool. Add entries freely; keep the
`## ` heading and the four labelled lines.

## Database connection pool exhausted
Category: database
Symptoms: connection timeout, pool exhausted, query latency in thousands of ms, upstream services return 500/502 shortly after
Confirm: db-service errors begin before errors in services that call it; pool usage at maximum
Fix: raise pool size or kill long-running queries; check for a slow query or connection leak introduced by the latest deploy

## Database disk full
Category: disk
Symptoms: no space left on device, write failed, writes rejected, reads still succeed
Confirm: errors appear only on write paths; disk usage at 100%
Fix: free space (rotate logs, drop old partitions), then expand the volume and add a disk usage alert at 80%

## Application memory leak
Category: memory
Symptoms: OutOfMemoryError, heap space, GC pause times growing, latency creeping up for minutes before failures, 503 from the service
Confirm: GC pauses and heap usage rise steadily before the crash; restarts temporarily fix it
Fix: restart the service as a stopgap, capture a heap dump, roll back the last deploy if the leak is new

## Authentication certificate or token failure
Category: auth
Symptoms: token validation failed, certificate expired, 401 responses, login failures, gateway auth checks failing
Confirm: failures are concentrated on auth-service and start at a clean boundary (often midnight or a deploy)
Fix: rotate or renew the signing certificate, restart auth-service, add certificate expiry monitoring

## Cache unavailable
Category: cache
Symptoms: connection refused on port 6379, cache miss storm, database load and latency rise, gateway slows down
Confirm: cache-service errors begin first, followed by db-service latency increase
Fix: restart or fail over the cache node; rate-limit cache-miss traffic to protect the database

## Network partition or upstream dependency down
Category: network
Symptoms: connection reset, host unreachable, DNS resolution failed, many services fail at the same moment
Confirm: multiple unrelated services fail in the same minute with network-type errors
Fix: check the load balancer, DNS and recent network changes; fail over to a healthy zone

## Bad deployment
Category: deploy
Symptoms: errors start exactly at a deploy time, new exception types never seen before, one service only
Confirm: compare error start time with the deploy log
Fix: roll back to the previous version, then debug in staging
