process SPLIT_PARQUET_BY_SAMPLE {
    tag "$meta.id"
    
    input:
    tuple val(meta), path(parquet_file)
    path(script)
    
    output:
    tuple val(meta), path("samplesheet.csv"), emit: samplesheet
    
    script:
    def args = task.ext.args ?: ''
    """
    python3 ${script} \\
        --input ${parquet_file} \\
        ${args}
    """
}
