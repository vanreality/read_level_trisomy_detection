process EXTRACT_READ_FROM_PARQUET {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(parquet_file), path(meth_file)
    path(max_coverage_file)
    val(mode)
    path(script)
    
    output:
    tuple val(meta), path("*.tsv"), emit: tsv
    
    script:
    def args = task.ext.args ?: ''
    def prefix = "${meta.id}_${meta.label}"
    """
    python3 ${script} \\
        --input ${parquet_file} \\
        --max_coverage ${max_coverage_file} \\
        --methylation ${meth_file} \\
        --mode ${mode} \\
        --prefix ${prefix} \\
        --ncpus ${task.cpus} \\
        ${args}
    """
}
