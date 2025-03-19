process CALCULATE_COVERAGE {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(parquet_file)
    path(script)
    
    output:
    tuple val(meta), path("*_coverage.csv"), emit: coverage
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --input ${parquet_file} \\
        --ncpus ${task.cpus} \\
        ${args}
    """
}
