process MERGE_COVERAGE {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(coverage_files)
    path(script)
    
    output:
    tuple val(meta), path("max_cov_pos.csv"), emit: merged_coverage
    
    script:
    def args = task.ext.args ?: ''
    def coverage_files_args = coverage_files.collect { "--input '$it'" }.join(' ')
    """
    python3 ${script} \\
        ${coverage_files_args} \\
        --ncpus ${task.cpus} \\
        ${args}
    """
}
