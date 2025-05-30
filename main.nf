// Custom analysis modules
include { SPLIT_PARQUET_BY_SAMPLE } from './modules/local/split_parquet_by_sample/main'
include { CALCULATE_COVERAGE } from './modules/local/calculate_coverage/main'
include { MERGE_COVERAGE } from './modules/local/merge_coverage/main'
include { EXTRACT_READ_FROM_PARQUET } from './modules/local/extract_read_from_parquet/main'

workflow  {
    // Define the channel for downstream processing
    if (params.input_parquet_samplesheet) {
        // Skip SPLIT_PARQUET_BY_SAMPLE and use provided samplesheet directly
        Channel
            .fromPath(params.input_parquet_samplesheet)
            .splitCsv(header: true)
            .map { row -> 
                def meta = [id: row.sample, label: row.label]
                // Check if parquet file exists
                def parquetFile = file(row.parquet)
                if (!parquetFile.exists()) {
                    error "Parquet file not found: ${row.parquet}"
                }
                return [meta, parquetFile]
            }
            .set { ch_parquet_samplesheet }
    } else {
        // 1. Split parquet by sample
        SPLIT_PARQUET_BY_SAMPLE(
                [[id: "sample"], file(params.input_parquet)],
                file("${workflow.projectDir}/bin/split_parquet_by_sample.py")
        )

        // Parse the generated samplesheet CSV to create channel
        SPLIT_PARQUET_BY_SAMPLE.out.samplesheet
            .map { meta, samplesheet -> samplesheet }
            .splitCsv(header: true)
            .map { row -> 
                def meta = [id: row.sample, label: row.label]
                // Check if parquet file exists
                def parquetFile = file(row.parquet)
                if (!parquetFile.exists()) {
                    error "Parquet file not found: ${row.parquet}"
                }
                return [meta, parquetFile]
            }
            .set { ch_parquet_samplesheet }
    }
    
    // Determine if we should use provided max_coverage_file or generate one
    if (params.max_coverage_file) {
        // Use the provided max coverage file as a value channel
        def maxCovFile = file(params.max_coverage_file)
        if (!maxCovFile.exists()) {
            error "Max coverage file not found: ${params.max_coverage_file}"
        }
        ch_max_coverage_file = Channel.value(maxCovFile)
    } else {
        // 2. Calculate max coverage positions of each sample
        CALCULATE_COVERAGE(
            ch_parquet_samplesheet,
            file("${workflow.projectDir}/bin/calculate_coverage.py")
        )
        
        // Group all coverage files together for merging
        CALCULATE_COVERAGE.out.coverage
            .map { meta, file -> file }
            .collect()
            .map { files -> [[id: "max_coverage"], files] }
            .set { ch_coverage_to_merge }
        
        // 3. Merge max coverage positions across samples
        MERGE_COVERAGE(
            ch_coverage_to_merge,
            file("${workflow.projectDir}/bin/merge_coverage.py")
        )
        
        MERGE_COVERAGE.out.merged_coverage
            .map { meta, file -> file }
            .set { ch_max_coverage_file }
    }
    
    ch_parquet_samplesheet
        .map { meta, parquetFile ->
            def methFile = file("${params.meth_dir}/${meta.id}_cpg_prob.csv")
            if (!methFile.exists()) {
                error "Methylation file not found: ${methFile}"
            }
            return [meta, parquetFile, methFile]
        }
        .set { ch_samplesheet }

    // 4. Extract reads from parquet
    EXTRACT_READ_FROM_PARQUET(
        ch_samplesheet,
        ch_max_coverage_file,
        params.mode,
        file("${workflow.projectDir}/bin/extract_read_from_parquet.py")
    )
}