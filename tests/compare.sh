#!/bin/bash
#Copyright 2026 Netherlands eScience Center
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


# Process positional arguments.
path1=$1
path2=$2

if [ -z "${path1}" ] || [ -z "${path2}" ]; then
    echo "compare.sh: missing operand after 'compare.sh'"
    exit 2
elif [ -f ${path1} ]; then
    if ! [ -f ${path2} ]; then
        echo "compare.sh: both paths must either be files or directories."
        exit 2
    fi
elif [ -d ${path1} ]; then
    if ! [ -d ${path2} ]; then
        echo "compare.sh: both paths must either be files or directories."
        exit 2
    fi
else
    echo "compare.sh: both paths must either be files or directories."
    exit 2
fi

# Compare either two files or directories.
# For directories, path2 may contain files that don't exist in path1,
# but all files that exist in path1 must exist and be the same in path2.
echo "Comparing ${path1} to ${path2}..."
out=$(diff ${path1} ${path2} | grep -v "Only in ${path2}:")
echo ${out}
echo "DONE"

if [ -z "${out}" ]; then exit 0; else exit 1; fi
