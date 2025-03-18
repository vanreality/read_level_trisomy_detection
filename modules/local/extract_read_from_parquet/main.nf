process EXTRACT_READ_FROM_PARQUET {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(parquet_file)
    path(max_coverage_file)
    val(mode)
    path(script)
    
    output:
    tuple val(meta), path("*.fa"), emit: fa
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --input ${parquet_file} \\
        --max_coverage ${max_coverage_file} \\
        --mode ${mode} \\
        ${args}
    """
}
