#!/bin/bash
#
# Description:
#    This script is used to check style on files that changed

changed_since_branching=$(git diff --name-only --diff-filter=ACM master... | tr " " "\n" | grep -E "*\.(cpp|hpp|h)")

exit_status=0

if [[ -n "${changed_since_branching}" ]]
then

    declare -i total
    total=0
    echo "Analyzing the following files: "
    echo
    for file in ${changed_since_branching}
    do
        declare -i changes
        changes=$(clang-format-6.0 -style=file -output-replacements-xml ${file} | grep -c "<replacement ")
        total+=changes
        echo "    ${file}: ${changes} changes needed to conform to style guide"
    done

    if [[ "${total}" > 0 ]]
    then
        exit_status=1
        echo
        echo "Apply changes using clang-format and resubmit"
    fi
else
    echo "No source file changed"
fi

exit ${exit_status}
