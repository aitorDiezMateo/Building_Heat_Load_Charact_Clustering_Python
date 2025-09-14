rule all:
    input:
        "output/02_Changepoint_Parameters_summary.csv",
        "output/03_Changepoint001.jpg",
        "output/03_outliers001.jpg",
        "output/03_outliers002.jpg",
        "output/03_outliers003.jpg",
        "output/03_outliers004.jpg",
        "output/03_processed_data.csv",
        "output/04_filled_data.csv",
        "output/05_repair001.jpg",
        "output/05_repair002.jpg",
        "output/06_final_processed_data.csv",
        "output/06_Changepoint_Pars_summ_TOW2.csv",
        "output/07_profiles001.jpg",
        "output/07_profiles002.jpg",
        "output/07_ElbowClust001.jpg",
        "output/07_SilhouetteClust001.jpg",
        "output/07_Cluster001.jpg",
        "output/07_ClusterProfiles_003.jpg",
        "output/07_cluster_assignments.csv",
        "output/09_confusion_matrix.csv",
        "output/09_classification_report.csv",
        "output/09_cart_model001.jpg",
        "output/09_cart_model002.jpg",
        "output/09_cart_model003.jpg",
        "output/09_cart_model.pkl",
        "output/09_Changepoint_Pars_summ_CLUST.csv",
        "output/09_Changepoint_Pars_summ_CLUST_PRED.csv",
        "output/10_model_metrics.csv",
        "output/10_Statistics_Graphics001.jpg",
        "output/10_Statistics_Graphics002.jpg",
        "output/10_Statistics_Graphics003.jpg",
        "output/10_Statistics_Graphics004.jpg",
        "output/10_Statistics_Graphics005.jpg",
        "output/10_Statistics_Graphics006.jpg",
        "output/10_Statistics_Graphics007.jpg",
        "output/10_Statistics_Graphics008.jpg",
        "output/10_Statistics_Graphics009.jpg",
        "output/10_Statistics_Graphics010.jpg"

rule load_data:
    input:
        "data/data.csv"
    output:
        "output/01_formatted.csv"
    shell:
        "C:/Users/aitor/anaconda3/envs/changepoint_env/python.exe scripts/01_load_data.py {input} {output}"

rule initial_changepoint_outliers:
    input:
        "output/01_formatted.csv"
    output:
        "output/02_Changepoint_Parameters_summary.csv"
    shell:
        "C:/Users/aitor/anaconda3/envs/changepoint_env/python.exe scripts/02_initial_changepoint_outliers.py {input} {output}"

rule inspection_changepoint_outliers:
    input:
        formatted="output/01_formatted.csv",
        summary="output/02_Changepoint_Parameters_summary.csv"
    output:
        changepoint="output/03_Changepoint001.jpg",
        out1="output/03_outliers001.jpg",
        out2="output/03_outliers002.jpg",
        out3="output/03_outliers003.jpg",
        out4="output/03_outliers004.jpg",
        processed="output/03_processed_data.csv"
    shell:
        "C:/Users/aitor/anaconda3/envs/changepoint_env/python.exe scripts/03_Inspection_Changepoint_Outliers.py {input.formatted} {input.summary} {output.changepoint} {output.out1} {output.out2} {output.out3} {output.out4} {output.processed}"

rule fill_data:
    input:
        "output/03_processed_data.csv"
    output:
        "output/04_filled_data.csv"
    shell:
        "C:/Users/aitor/anaconda3/envs/changepoint_env/python.exe scripts/04_Fill_Data.py {input} {output}"

rule inspection_fill_data:
    input:
        "output/04_filled_data.csv"
    output:
        plot1="output/05_repair001.jpg",
        plot2="output/05_repair002.jpg"
    shell:
        "C:/Users/aitor/anaconda3/envs/changepoint_env/python.exe scripts/05_Inspection_Fill_Data.py {input} {output.plot1} {output.plot2}"

rule final_changepoint_model:
    input:
        "output/04_filled_data.csv"
    output:
        data="output/06_final_processed_data.csv",
        params="output/06_Changepoint_Pars_summ_TOW2.csv"
    shell:
        "C:/Users/aitor/anaconda3/envs/changepoint_env/python.exe scripts/06_Final_Changepoint_Model.py {input} {output.data} {output.params}"

rule clusterization:
    input:
        "output/06_final_processed_data.csv"
    output:
        profiles1="output/07_profiles001.jpg",
        profiles2="output/07_profiles002.jpg",
        elbow="output/07_ElbowClust001.jpg",
        silhouette="output/07_SilhouetteClust001.jpg",
        cluster="output/07_Cluster001.jpg",
        comprehensive="output/07_ClusterProfiles_003.jpg",
        assignments="output/07_cluster_assignments.csv"
    shell:
        "C:/Users/aitor/anaconda3/envs/changepoint_env/python.exe scripts/07_Clusterization.py {input} {output.profiles1} {output.profiles2} {output.elbow} {output.silhouette} {output.cluster} {output.comprehensive} {output.assignments}"

rule cart:
    input:
        data="output/06_final_processed_data.csv",
        clusters="output/07_cluster_assignments.csv"
    output:
        confusion="output/09_confusion_matrix.csv",
        tree="output/09_cart_model001.jpg",
        comparison="output/09_cart_model002.jpg",
        accuracy="output/09_cart_model003.jpg",
        report_csv="output/09_classification_report.csv",
        model="output/09_cart_model.pkl"
    shell:
        "C:/Users/aitor/anaconda3/envs/changepoint_env/python.exe scripts/08_CART.py {input.data} {input.clusters} {output.confusion} {output.tree} {output.comparison} {output.accuracy} {output.report_csv} {output.model}"

rule changepoint_clusters:
    input:
        data="output/06_final_processed_data.csv",
        clusters="output/07_cluster_assignments.csv"
    output:
        clust_params="output/09_Changepoint_Pars_summ_CLUST.csv",
        clust_pred_params="output/09_Changepoint_Pars_summ_CLUST_PRED.csv"
    shell:
        "C:/Users/aitor/anaconda3/envs/changepoint_env/python.exe scripts/09_Changepoint_Clusters.py {input.data} {input.clusters} {output.clust_params} {output.clust_pred_params}"

rule statistics_graphics:
    input:
        data="output/06_final_processed_data.csv",
        clusters="output/07_cluster_assignments.csv",
        clust_params="output/09_Changepoint_Pars_summ_CLUST.csv",
        clust_pred_params="output/09_Changepoint_Pars_summ_CLUST_PRED.csv"
    output:
        metrics="output/10_model_metrics.csv",
        plot1="output/10_Statistics_Graphics001.jpg",
        plot2="output/10_Statistics_Graphics002.jpg",
        plot3="output/10_Statistics_Graphics003.jpg",
        plot4="output/10_Statistics_Graphics004.jpg",
        plot5="output/10_Statistics_Graphics005.jpg",
        plot6="output/10_Statistics_Graphics006.jpg",
        plot7="output/10_Statistics_Graphics007.jpg",
        plot8="output/10_Statistics_Graphics008.jpg",
        plot9="output/10_Statistics_Graphics009.jpg",
        plot10="output/10_Statistics_Graphics010.jpg"
    shell:
        "C:/Users/aitor/anaconda3/envs/changepoint_env/python.exe scripts/10_Statistics_Graphics.py {input.data} {input.clusters} {input.clust_params} {input.clust_pred_params} {output.metrics} {output.plot1} {output.plot2} {output.plot3} {output.plot4} {output.plot5} {output.plot6} {output.plot7} {output.plot8} {output.plot9} {output.plot10}"