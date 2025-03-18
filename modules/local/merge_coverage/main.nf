process MERGE_COVERAGE {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(coverage_files)
    path(script)
    
    output:
    tuple val(meta), path("merged_coverage.csv"), emit: merged_coverage
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --input ${coverage_files} \\
        ${args}
    """
}
