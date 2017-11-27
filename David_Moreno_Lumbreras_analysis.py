#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:44:08 2017

@author: David Moreno
"""
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import glob
import time
import datetime
from datetime import timedelta
from sklearn.cluster import MiniBatchKMeans, KMeans,  MeanShift, estimate_bandwidth
from itertools import cycle
from sklearn.metrics.pairwise import pairwise_distances_argmin

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

def load_acc_data(path, tab):
    xl = pd.ExcelFile(path)
    df = xl.parse(tab)
    return df

def load_speed_data():
    path ='speed_total/' # use your path
    allFiles = glob.glob(path + "/*.txt")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_table(file_,index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    frame["5 Minutes"] = pd.to_datetime(frame["5 Minutes"], format='%m/%d/%Y %H:%M', errors='coerce')
    return frame

def load_occ_data():
    path ='occ_total/' # use your path
    allFiles = glob.glob(path + "/*.txt")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_table(file_,index_col=None, header=0)
        list_.append(df)
    frame = pd.concat(list_)
    frame["5 Minutes"] = pd.to_datetime(frame["5 Minutes"], format='%m/%d/%Y %H:%M', errors='ignore')
    return frame

def delete_extract_columns(X): 
    #Sacar dias de la semana de findes, eliminar columnas que no dan mucha informacion
    rm_cols = ["Unnamed: 34", "Unnamed: 33", "cnty_rte", "hwy_grp", "int_rmp"]
    
    X.drop(rm_cols, axis=1, inplace=True)
    
    #cause1 = accident cause
    X["cause1"] = X["cause1"].astype('category')
    #acctype = type of colission
    X["acctype"] = X["acctype"].astype('category')
    #inter = intersection crash
    X["inter"] = X["inter"].astype('category')
    #weather1 and 2 IMPORTANT = weather
    X["weather1"] = X["weather1"].astype('category')
    X["weather2"] = X["weather2"].astype('category')
    #loc_typ = location type (interseccion, highway, ramp)
    X["loc_typ"] = X["loc_typ"].astype('category')
    #towaway = injury, fatal ...
    X["towaway"] = X["towaway"].astype('category')
    #veh_invl = motor vehicules
    X["veh_invl"] = X["veh_invl"].astype('category')
    #rdsurf = road surface
    X["rdsurf"] = X["rdsurf"].astype('category')
    #rd_def1 and def2 = road condition
    X["rd_def1"] = X["rd_def1"].astype('category')
    X["rd_def2"] = X["rd_def2"].astype('category')
    #light = light condition
    X["light"] = X["light"].astype('category')
    #trak_flg = truck involved
    X["trk_flg"] = X["trk_flg"].astype('category')
    
    
    #Convert categoryc into numeric
    cat_columns = X.select_dtypes(['category']).columns
    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
    
    #Date and hour
    X["acc_date"] = pd.to_datetime(X["acc_date"], format='%Y%m%d', errors='ignore');
    for i, row in X.iterrows():
        ifor_val = row["caseno"][:12]
        X.set_value(i,'caseno',ifor_val)
    X["caseno"] = pd.to_datetime(X["caseno"], format='%Y%m%d%H%M', errors='coerce');

    #Alameda country and correct milepost
    X_acc_SF = X[X["county"] == 1]
    X_acc_SF = X_acc_SF[X_acc_SF["milepost"] >= 3.75]
    X_acc_SF =  X_acc_SF[X_acc_SF["milepost"] <= 5.25]
    X_acc_SF = X_acc_SF.reset_index(drop=True)
    

    return X, X_acc_SF

#PCA for accidents
def analyze_pca(X): 
    pca = PCA(n_components=2)
    pca.fit(X)
    
    return pca

#Not work
def join_dataframes(df, acc): 
    #Primero añadir las columnas
    df["severity"] = 0
    df["accident"] = False
    for i, row in df.iterrows():
        print(i)
        data = acc[acc["caseno"] == row["5 Minutes"]]
        if len(data["severity"]) > 0:
            df.set_value(i,'severity',data["severity"])
            df.set_value(i,'accident',True)
        
    return df

def accidents_vs_occ_severity_one_more(occ, acc):
    #So, we are going to show what happen with the occupation one hour after and before of an accident of more than severity one
    for i in range(len(acc)):
        if acc['severity'][i] > 1:
            aux = occ[occ["5 Minutes"] > acc['caseno'][i]-timedelta(hours=1)]
            aux = aux[aux["5 Minutes"] < acc['caseno'][i]+timedelta(hours=1)]
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            l1, l2, l3, l4, l5, lmean = ax1.plot( aux["5 Minutes"], aux["Lane 1 Occ (%)"], 'r', aux["5 Minutes"], aux["Lane 2 Occ (%)"], 'b', aux["5 Minutes"], aux["Lane 3 Occ (%)"], 'g', aux["5 Minutes"], aux["Lane 4 Occ (%)"], 'y', aux["5 Minutes"], aux["Lane 5 Occ (%)"], 'c', aux["5 Minutes"], aux["Occupancy (%)"], 'm')
            ax1.set_title("Accident on: " + acc['caseno'][i].strftime("%B %d, %Y - %H:%M"))
            ax1.set_ylabel('Occupation')
            ax1.set_xlabel('Hour')
            fig1.legend((l1, l2, l3, l4, l5, lmean), ('Lane 1', 'Lane 2', 'Lane 3', 'Lane 4', 'Lane 5', 'Mean'), 'upper right')
    
            ax2 = ax1.twinx()
            ax2.plot(acc["caseno"][i], acc["severity"][i], 'og')
            ax2.set_ylabel('severity', color='g')
            ax2.tick_params('y', colors='g')
    
def accidents_vs_speed_severity_one_more(speed, acc):
    #So, we are going to show what happen with the speed one hour after and before for an accident of more than severity one
    for i in range(len(acc)):
        if acc['severity'][i] > 1:
            aux = speed[speed["5 Minutes"] > acc['caseno'][i]-timedelta(hours=1)]
            aux = aux[aux["5 Minutes"] < acc['caseno'][i]+timedelta(hours=1)]
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            l1, l2, l3, l4, l5, lmean = ax1.plot( aux["5 Minutes"], aux["Lane 1 Speed (mph)"], 'r', aux["5 Minutes"], aux["Lane 2 Speed (mph)"], 'b', aux["5 Minutes"], aux["Lane 3 Speed (mph)"], 'g', aux["5 Minutes"], aux["Lane 4 Speed (mph)"], 'y', aux["5 Minutes"], aux["Lane 5 Speed (mph)"], 'c', aux["5 Minutes"], aux["Speed (mph)"], 'm')
            ax1.set_title("Accident on: " + acc['caseno'][i].strftime("%B %d, %Y - %H:%M"))
            ax1.set_ylabel('Speed')
            ax1.set_xlabel('Hour')
            fig1.legend((l1, l2, l3, l4, l5, lmean), ('Lane 1', 'Lane 2', 'Lane 3', 'Lane 4', 'Lane 5', 'Mean'), 'upper right')
    
            ax2 = ax1.twinx()
            ax2.plot(acc["caseno"][i], acc["severity"][i], 'og')
            ax2.set_ylabel('severity', color='g')
            ax2.tick_params('y', colors='g')
            
def accidents_vs_flow_severity_one_more(speed, acc):
    #So, we are going to show what happen with the speed one hour after and before of an accident of more than severity one
    for i in range(len(acc)):
        if acc['severity'][i] > 1:
            aux = speed[speed["5 Minutes"] > acc['caseno'][i]-timedelta(hours=1)]
            aux = aux[aux["5 Minutes"] < acc['caseno'][i]+timedelta(hours=1)]
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            l1, l2, l3, l4, l5, lmean = ax1.plot( aux["5 Minutes"], aux["Lane 1 Flow (Veh/5 Minutes)"], 'r', aux["5 Minutes"], aux["Lane 2 Flow (Veh/5 Minutes)"], 'b', aux["5 Minutes"], aux["Lane 3 Flow (Veh/5 Minutes)"], 'g', aux["5 Minutes"], aux["Lane 4 Flow (Veh/5 Minutes)"], 'y', aux["5 Minutes"], aux["Lane 5 Flow (Veh/5 Minutes)"], 'c', aux["5 Minutes"], aux["Flow (Veh/5 Minutes)"], 'm')
            ax1.set_title("Accident on: " + acc['caseno'][i].strftime("%B %d, %Y - %H:%M"))
            ax1.set_ylabel('Flow')
            ax1.set_xlabel('Hour')
            fig1.legend((l1, l2, l3, l4, l5, lmean), ('Lane 1', 'Lane 2', 'Lane 3', 'Lane 4', 'Lane 5', 'Total'), 'upper right')
    
            ax2 = ax1.twinx()
            ax2.plot(acc["caseno"][i], acc["severity"][i], 'og')
            ax2.set_ylabel('severity', color='g')
            ax2.tick_params('y', colors='g')
            
def density_vs_flow_lanes(occ, speed, acc):
    density = pd.DataFrame()
    density["5 Minutes"] = occ["5 Minutes"]
    
    density["density"] = occ["Lane 1 Flow (Veh/5 Minutes)"]/speed["Lane 1 Speed (mph)"]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot( density["5 Minutes"], density["density"], 'b')
    ax1.set_title("Density on Lane 1")
    ax1.set_ylabel('Density')
    ax1.set_xlabel('Day')
    ax2 = ax1.twinx()
    ax2.plot(acc["caseno"], acc["severity"], 'og')
    ax2.set_ylabel('Accident severity', color='g')
    ax2.tick_params('y', colors='g')
    
    density["density"] = occ["Lane 2 Flow (Veh/5 Minutes)"]/speed["Lane 2 Speed (mph)"]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot( density["5 Minutes"], density["density"], 'b')
    ax1.set_title("Density on Lane 2")
    ax1.set_ylabel('Density')
    ax1.set_xlabel('Day')
    ax2 = ax1.twinx()
    ax2.plot(acc["caseno"], acc["severity"], 'og')
    ax2.set_ylabel('Accident severity', color='g')
    ax2.tick_params('y', colors='g')
    
    density["density"] = occ["Lane 3 Flow (Veh/5 Minutes)"]/speed["Lane 3 Speed (mph)"]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot( density["5 Minutes"], density["density"], 'b')
    ax1.set_title("Density on Lane 3")
    ax1.set_ylabel('Density')
    ax1.set_xlabel('Day')
    ax2 = ax1.twinx()
    ax2.plot(acc["caseno"], acc["severity"], 'og')
    ax2.set_ylabel('Accident severity', color='g')
    ax2.tick_params('y', colors='g')
    
    density["density"] = occ["Lane 4 Flow (Veh/5 Minutes)"]/speed["Lane 4 Speed (mph)"]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot( density["5 Minutes"], density["density"], 'b')
    ax1.set_title("Density on Lane 4")
    ax1.set_ylabel('Density')
    ax1.set_xlabel('Day')
    ax2 = ax1.twinx()
    ax2.plot(acc["caseno"], acc["severity"], 'og')
    ax2.set_ylabel('Accident severity', color='g')
    ax2.tick_params('y', colors='g')
    
    density["density"] = occ["Lane 5 Flow (Veh/5 Minutes)"]/speed["Lane 5 Speed (mph)"]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot( density["5 Minutes"], density["density"], 'b')
    ax1.set_title("Density on Lane 5")
    ax1.set_ylabel('Density')
    ax1.set_xlabel('Day')
    ax2 = ax1.twinx()
    ax2.plot(acc["caseno"], acc["severity"], 'og')
    ax2.set_ylabel('Accident severity', color='g')
    ax2.tick_params('y', colors='g')
    
def density_vs_flow_cluster(occ, speed, acc):
    density = pd.DataFrame()
    density["5 Minutes"] = occ["5 Minutes"]
    density["density"] = occ["Flow (Veh/5 Minutes)"]/speed["Speed (mph)"]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot( occ["Flow (Veh/5 Minutes)"], density["density"], 'ob')
    ax1.set_title("Density vs total flow")
    ax1.set_ylabel('Density')
    ax1.set_xlabel('Flow')
    
    density_flow = pd.DataFrame()
    density_flow["density"] = density["density"]
    density_flow["flow"] = occ["Flow (Veh/5 Minutes)"]
    
    
    #Clustering
    arr = density_flow[['density', 'flow']].values
    
    kmeans_and_miniBatchKMeans(arr)
    
    meanshift_cluster(arr)
    
    agglomerative_cluster_connectivity(arr)

    

    

    
def kmeans_and_miniBatchKMeans(arr):
    batch_size = 100
    n_clusters = 3
    # Compute clustering with Means
    
    k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
    t0 = time.time()
    k_means.fit(arr)
    t_batch = time.time() - t0
    
    # Compute clustering with MiniBatchKMeans
    
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(arr)
    t_mini_batch = time.time() - t0
    # Plot result
    
    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    
    # We want to have the same colors for the same cluster from the
    # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
    # closest one.
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(arr, k_means_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(arr, mbk_means_cluster_centers)
    order = pairwise_distances_argmin(k_means_cluster_centers,
                                      mbk_means_cluster_centers)
    
    # KMeans
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(arr[my_members, 0], arr[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
        t_batch, k_means.inertia_))
    
    # MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = mbk_means_labels == order[k]
        cluster_center = mbk_means_cluster_centers[order[k]]
        ax.plot(arr[my_members, 0], arr[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    ax.set_title('MiniBatchKMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
             (t_mini_batch, mbk.inertia_))
    
    # Initialise the different array to all False
    different = (mbk_means_labels == 4)
    ax = fig.add_subplot(1, 3, 3)
    
    for k in range(n_clusters):
        different += ((k_means_labels == k) != (mbk_means_labels == order[k]))
    
    identic = np.logical_not(different)
    ax.plot(arr[identic, 0], arr[identic, 1], 'w',
            markerfacecolor='#bbbbbb', marker='.')
    ax.plot(arr[different, 0], arr[different, 1], 'w',
            markerfacecolor='m', marker='.')
    ax.set_title('Difference')
    ax.set_xticks(())
    ax.set_yticks(())
    
    plt.show()
    
def meanshift_cluster(arr):
    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(arr, quantile=0.2, n_samples=500)
    
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(arr)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    
    # Plot result
    
    plt.figure(2)
    plt.clf()
    
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(arr[my_members, 0], arr[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('MeanShift: estimated number of clusters: %d' % n_clusters_)
    plt.show()
    
def agglomerative_cluster_connectivity(arr):
    knn_graph = kneighbors_graph(arr, 10, include_self=False)

    for connectivity in (None, knn_graph):
        for n_clusters in (10, 3):
            plt.figure(figsize=(10, 4))
            for index, linkage in enumerate(('average', 'complete', 'ward')):
                plt.subplot(1, 3, index + 1)
                model = AgglomerativeClustering(linkage=linkage,
                                                connectivity=connectivity,
                                                n_clusters=n_clusters)
                t0 = time.time()
                model.fit(arr)
                elapsed_time = time.time() - t0
                plt.scatter(arr[:, 0], arr[:, 1], c=model.labels_,
                            cmap=plt.cm.spectral)
                plt.title('linkage=%s (time %.2fs)' % (linkage, elapsed_time),
                          fontdict=dict(verticalalignment='top'))
                plt.axis('equal')
                plt.axis('off')
    
                plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                                    left=0, right=1)
                plt.suptitle('n_cluster=%i, connectivity=%r' %
                             (n_clusters, connectivity is not None), size=17)
    
    
    plt.show()

if __name__ == "__main__":
    #Load data
    X = load_acc_data("HSIS _2006_ crash_data.xlsx", "2006 I80 E");
    occ_df = load_occ_data();
    speed_df = load_speed_data();

    #Only analysis SF and the correct milepost
    X, X_acc_SF = delete_extract_columns(X);
    
   
    #Only analysis in November
    occ_df = occ_df[(occ_df['5 Minutes']>datetime.date(2006,11,1)) & (occ_df['5 Minutes']<datetime.date(2006,11,30))]
    speed_df = speed_df[(speed_df['5 Minutes']>datetime.date(2006,11,1)) & (speed_df['5 Minutes']<datetime.date(2006,11,30))]
    X_acc_SF = X_acc_SF[(X_acc_SF['caseno']>datetime.date(2006,11,1)) & (X_acc_SF['caseno']<datetime.date(2006,11,30))]
    X_acc_SF = X_acc_SF.reset_index(drop=True)
    occ_df = occ_df.reset_index(drop=True)
    speed_df = speed_df.reset_index(drop=True)
    
    #What happen with occ, speed and flow with an accident
    accidents_vs_occ_severity_one_more(occ_df, X_acc_SF);
    accidents_vs_speed_severity_one_more(speed_df, X_acc_SF)
    accidents_vs_flow_severity_one_more(speed_df, X_acc_SF)
    
    #Density vs flow
    density_vs_flow_lanes(occ_df, speed_df, X_acc_SF)
    
    #Clustering of de density vs flow
    density_vs_flow_cluster(occ_df, speed_df, X_acc_SF)
    
    
    
    plt.show()

