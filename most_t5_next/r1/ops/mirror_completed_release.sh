#!/usr/bin/env bash
# Mirror only immutable completed production artifacts to the retained region.
# This script is intentionally append/copy-only: it never deletes source or
# destination content and excludes in-progress shard attempt directories.

set -eu

if [ "$#" -ne 6 ]; then
  printf '%s\n' "usage: $0 SOURCE_ROOT DEST_HOST DEST_PORT IDENTITY_FILE DEST_ROOT INTERVAL_SECONDS" >&2
  exit 2
fi

source_root=${1%/}
destination_host=$2
destination_port=$3
identity_file=$4
destination_root=${5%/}
interval_seconds=$6

if [ ! -d "$source_root" ]; then
  printf '%s\n' "source release root is not a directory" >&2
  exit 2
fi
if [ ! -f "$identity_file" ]; then
  printf '%s\n' "SSH identity file is not a regular file" >&2
  exit 2
fi
case "$destination_port" in
  ''|*[!0-9]*) printf '%s\n' "destination port must be numeric" >&2; exit 2 ;;
esac
case "$interval_seconds" in
  ''|*[!0-9]*) printf '%s\n' "interval must be numeric" >&2; exit 2 ;;
esac

remote_shell="ssh -i $identity_file -p $destination_port -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
destination="root@$destination_host:$destination_root/"
source_manifest="$source_root/full_release_manifest.json"
destination_manifest="$destination_root/full_release_manifest.json"

quote_for_remote_shell() {
  # Single-quote one path for the remote POSIX shell.  The replacement emits
  # the standard '\'' sequence for any embedded single quote.
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

destination_manifest_quoted=$(quote_for_remote_shell "$destination_manifest")

is_sha256() {
  value=$1
  [ "${#value}" -eq 64 ] || return 1
  case "$value" in
    *[!0-9a-f]*) return 1 ;;
  esac
}

local_manifest_sha256() {
  output=$(sha256sum -- "$1") || return 1
  value=${output%% *}
  is_sha256 "$value" || return 1
  printf '%s\n' "$value"
}

remote_manifest_sha256() {
  output=$(ssh -i "$identity_file" -p "$destination_port" \
    -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    "root@$destination_host" "sha256sum -- $destination_manifest_quoted") || return 1
  value=${output%% *}
  is_sha256 "$value" || return 1
  printf '%s\n' "$value"
}

while :; do
  # Only exit after a manifest that existed before rsync built its file list
  # has participated in a successful pass.  Testing for the manifest only
  # after rsync races with the producer: a newly-created manifest could make
  # the loop exit even though that rsync invocation never saw it.
  release_was_complete_before_rsync=0
  source_manifest_sha256_before=''
  if [ -f "$source_manifest" ]; then
    if source_manifest_sha256_before=$(local_manifest_sha256 "$source_manifest"); then
      release_was_complete_before_rsync=1
    fi
  fi
  if rsync -a --partial \
      --exclude='shard-*.partial-attempt-*' \
      --exclude='.run_state.*.tmp' \
      --exclude='.write-json.*.tmp' \
      -e "$remote_shell" \
      "$source_root/" "$destination"; then
    completed=$(find "$source_root" -mindepth 1 -maxdepth 1 -type d \
      -regextype posix-extended -regex '.*/shard-[0-9]{6}' | wc -l)
    # The count is sampled on the source after rsync returns.  Shards created
    # after rsync built its file list will be copied on the next iteration, so
    # do not mislabel this observation as a destination-side count.
    printf '{"source_completed_shards_seen_after_successful_rsync":%s,"status":"pass","utc":"%s"}\n' \
      "$completed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ "$release_was_complete_before_rsync" -eq 1 ]; then
      source_manifest_sha256_after=''
      destination_manifest_sha256=''
      if [ -f "$source_manifest" ] && \
          source_manifest_sha256_after=$(local_manifest_sha256 "$source_manifest") && \
          [ "$source_manifest_sha256_before" = "$source_manifest_sha256_after" ] && \
          destination_manifest_sha256=$(remote_manifest_sha256) && \
          [ "$destination_manifest_sha256" = "$source_manifest_sha256_after" ]; then
        exit 0
      fi
    fi
  else
    status=$?
    printf '{"rsync_exit":%s,"status":"retry","utc":"%s"}\n' \
      "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
  fi
  sleep "$interval_seconds"
done
