SAMPLES = ["data"]
SLURM_CLUSTER = False
import sys
PATH_TO_PYTHON = sys.executable

rule all:
    input:
        expand("output/{sample}_02_Changepoint_Parameters_summary.csv", sample=SAMPLES),
        expand("output/{sample}_03_Changepoint001.jpg", sample=SAMPLES),
        expand("output/{sample}_03_outliers001.jpg", sample=SAMPLES),
        expand("output/{sample}_03_outliers002.jpg", sample=SAMPLES),
        expand("output/{sample}_03_outliers003.jpg", sample=SAMPLES),
        expand("output/{sample}_03_outliers004.jpg", sample=SAMPLES),
        expand("output/{sample}_03_processed_data.csv", sample=SAMPLES),
        expand("output/{sample}_04_filled_data.csv", sample=SAMPLES),
        expand("output/{sample}_05_repair001.jpg", sample=SAMPLES),
        expand("output/{sample}_05_repair002.jpg", sample=SAMPLES),
        expand("output/{sample}_06_final_processed_data.csv", sample=SAMPLES),
        expand("output/{sample}_06_Changepoint_Pars_summ_TOW2.csv", sample=SAMPLES),
        expand("output/{sample}_07_profiles001.jpg", sample=SAMPLES),
        expand("output/{sample}_07_profiles002.jpg", sample=SAMPLES),
        expand("output/{sample}_07_ElbowClust001.jpg", sample=SAMPLES),
        expand("output/{sample}_07_SilhouetteClust001.jpg", sample=SAMPLES),
        expand("output/{sample}_07_Cluster001.jpg", sample=SAMPLES),
        expand("output/{sample}_07_ClusterProfiles_003.jpg", sample=SAMPLES),
        expand("output/{sample}_07_cluster_assignments.csv", sample=SAMPLES),
        expand("output/{sample}_09_confusion_matrix.csv", sample=SAMPLES),
        expand("output/{sample}_09_classification_report.csv", sample=SAMPLES),
        expand("output/{sample}_09_cart_model001.jpg", sample=SAMPLES),
        expand("output/{sample}_09_cart_model002.jpg", sample=SAMPLES),
        expand("output/{sample}_09_cart_model003.jpg", sample=SAMPLES),
        expand("output/{sample}_09_cart_model.pkl", sample=SAMPLES),
        expand("output/{sample}_09_Changepoint_Pars_summ_CLUST.csv", sample=SAMPLES),
        expand("output/{sample}_09_Changepoint_Pars_summ_CLUST_PRED.csv", sample=SAMPLES),
        expand("output/{sample}_10_model_metrics.csv", sample=SAMPLES),
        expand("output/{sample}_10_Statistics_Graphics001.jpg", sample=SAMPLES),
        expand("output/{sample}_10_Statistics_Graphics002.jpg", sample=SAMPLES),
        expand("output/{sample}_10_Statistics_Graphics003.jpg", sample=SAMPLES),
        expand("output/{sample}_10_Statistics_Graphics004.jpg", sample=SAMPLES),
        expand("output/{sample}_10_Statistics_Graphics005.jpg", sample=SAMPLES),
        expand("output/{sample}_10_Statistics_Graphics006.jpg", sample=SAMPLES),
        expand("output/{sample}_10_Statistics_Graphics007.jpg", sample=SAMPLES),
        expand("output/{sample}_10_Statistics_Graphics008.jpg", sample=SAMPLES),
        expand("output/{sample}_10_Statistics_Graphics009.jpg", sample=SAMPLES),
        expand("output/{sample}_10_Statistics_Graphics010.jpg", sample=SAMPLES)

rule load_data:
    input:
        data="data/{sample}.csv",
        weather="data/weather_irradiation.csv"
    output:
        "output/{sample}_01_formatted.csv"
    threads: 1
    shell:
        "{PATH_TO_PYTHON} scripts/01_load_data.py {input.data} {input.weather} {output}"

rule initial_changepoint_outliers:
    input:
        "output/{sample}_01_formatted.csv"
    output:
        "output/{sample}_02_Changepoint_Parameters_summary.csv"
    threads: 8
    params:
        use_slurm=SLURM_CLUSTER
    shell:
        "{PATH_TO_PYTHON} scripts/02_initial_changepoint_outliers.py {input} {output} {params.use_slurm}"

rule inspection_changepoint_outliers:
    input:
        formatted="output/{sample}_01_formatted.csv",
        summary="output/{sample}_02_Changepoint_Parameters_summary.csv"
    output:
        changepoint="output/{sample}_03_Changepoint001.jpg",
        out1="output/{sample}_03_outliers001.jpg",
        out2="output/{sample}_03_outliers002.jpg",
        out3="output/{sample}_03_outliers003.jpg",
        out4="output/{sample}_03_outliers004.jpg",
        processed="output/{sample}_03_processed_data.csv"
    threads: 1
    shell:
        "{PATH_TO_PYTHON} scripts/03_Inspection_Changepoint_Outliers.py {input.formatted} {input.summary} {output.changepoint} {output.out1} {output.out2} {output.out3} {output.out4} {output.processed}"

rule fill_data:
    input:
        "output/{sample}_03_processed_data.csv"
    output:
        "output/{sample}_04_filled_data.csv"
    threads: 1
    shell:
        "{PATH_TO_PYTHON} scripts/04_Fill_Data.py {input} {output}"

rule inspection_fill_data:
    input:
        "output/{sample}_04_filled_data.csv"
    output:
        plot1="output/{sample}_05_repair001.jpg",
        plot2="output/{sample}_05_repair002.jpg"
    threads: 1
    shell:
        "{PATH_TO_PYTHON} scripts/05_Inspection_Fill_Data.py {input} {output.plot1} {output.plot2}"

rule final_changepoint_model:
    input:
        "output/{sample}_04_filled_data.csv"
    output:
        data="output/{sample}_06_final_processed_data.csv",
        params="output/{sample}_06_Changepoint_Pars_summ_TOW2.csv"
    threads: 8
    params:
        use_slurm=SLURM_CLUSTER
    shell:
        "{PATH_TO_PYTHON} scripts/06_Final_Changepoint_Model.py {input} {output.data} {output.params} {params.use_slurm}"

rule clusterization:
    input:
        "output/{sample}_06_final_processed_data.csv"
    output:
        profiles1="output/{sample}_07_profiles001.jpg",
        profiles2="output/{sample}_07_profiles002.jpg",
        elbow="output/{sample}_07_ElbowClust001.jpg",
        silhouette="output/{sample}_07_SilhouetteClust001.jpg",
        cluster="output/{sample}_07_Cluster001.jpg",
        comprehensive="output/{sample}_07_ClusterProfiles_003.jpg",
        assignments="output/{sample}_07_cluster_assignments.csv"
    threads: 1
    shell:
        "{PATH_TO_PYTHON} scripts/07_Clusterization.py {input} {output.profiles1} {output.profiles2} {output.elbow} {output.silhouette} {output.cluster} {output.comprehensive} {output.assignments}"

rule cart:
    input:
        data="output/{sample}_06_final_processed_data.csv",
        clusters="output/{sample}_07_cluster_assignments.csv"
    output:
        confusion="output/{sample}_09_confusion_matrix.csv",
        tree="output/{sample}_09_cart_model001.jpg",
        comparison="output/{sample}_09_cart_model002.jpg",
        accuracy="output/{sample}_09_cart_model003.jpg",
        report_csv="output/{sample}_09_classification_report.csv",
        model="output/{sample}_09_cart_model.pkl"
    threads: 1
    shell:
        "{PATH_TO_PYTHON} scripts/08_CART.py {input.data} {input.clusters} {output.confusion} {output.tree} {output.comparison} {output.accuracy} {output.report_csv} {output.model}"

rule changepoint_clusters:
    input:
        data="output/{sample}_06_final_processed_data.csv",
        clusters="output/{sample}_07_cluster_assignments.csv"
    output:
        clust_params="output/{sample}_09_Changepoint_Pars_summ_CLUST.csv",
        clust_pred_params="output/{sample}_09_Changepoint_Pars_summ_CLUST_PRED.csv"
    threads: 16
    params:
        use_slurm=SLURM_CLUSTER
    shell:
        "{PATH_TO_PYTHON} scripts/09_Changepoint_Clusters.py {input.data} {input.clusters} {output.clust_params} {output.clust_pred_params} {params.use_slurm}"

rule statistics_graphics:
    input:
        data="output/{sample}_06_final_processed_data.csv",
        clusters="output/{sample}_07_cluster_assignments.csv",
        clust_params="output/{sample}_09_Changepoint_Pars_summ_CLUST.csv",
        clust_pred_params="output/{sample}_09_Changepoint_Pars_summ_CLUST_PRED.csv"
    output:
        metrics="output/{sample}_10_model_metrics.csv",
        plot1="output/{sample}_10_Statistics_Graphics001.jpg",
        plot2="output/{sample}_10_Statistics_Graphics002.jpg",
        plot3="output/{sample}_10_Statistics_Graphics003.jpg",
        plot4="output/{sample}_10_Statistics_Graphics004.jpg",
        plot5="output/{sample}_10_Statistics_Graphics005.jpg",
        plot6="output/{sample}_10_Statistics_Graphics006.jpg",
        plot7="output/{sample}_10_Statistics_Graphics007.jpg",
        plot8="output/{sample}_10_Statistics_Graphics008.jpg",
        plot9="output/{sample}_10_Statistics_Graphics009.jpg",
        plot10="output/{sample}_10_Statistics_Graphics010.jpg"
    threads: 1
    shell:
        "{PATH_TO_PYTHON} scripts/10_Statistics_Graphics.py {input.data} {input.clusters} {input.clust_params} {input.clust_pred_params} {output.metrics} {output.plot1} {output.plot2} {output.plot3} {output.plot4} {output.plot5} {output.plot6} {output.plot7} {output.plot8} {output.plot9} {output.plot10}"