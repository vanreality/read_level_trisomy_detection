process MERGE_COVERAGE {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(coverage_files)
    path(script)
    
    output:
    tuple val(meta), path("max_cov_pos.csv"), emit: merged_coverage
    
    script:
    def args = task.ext.args ?: ''
    def coverage_files_str = coverage_files.join(' ')
    """
    python3 ${script} \\
        --input ${coverage_files_str} \\
        ${args}
    """
}
