# DO NOT change anything except within the function
from approvedimports import *

def cluster_and_visualise(datafile_name:str, K:int, feature_names:list):
    """Function to get the data from a file, perform K-means clustering and produce a visualisation of results.

    Parameters
        ----------
        datafile_name: str
            path to data file

        K: int
            number of clusters to use

        feature_names: list
            list of feature names

        Returns
        ---------
        fig: matplotlib.figure.Figure
            the figure object for the plot

        axs: matplotlib.axes.Axes
            the axes object for the plot
    """
   # ====> insert your code below here
    # get the data from file into a numpy array
    data = np.genfromtxt(datafile_name, delimiter=',')


    cluster_model = KMeans(n_clusters=K, n_init=10)
    cluster_model.fit(data)
    cluster_ids = cluster_model.predict(data)

    # Find the number of features
    num_feat = data.shape[1]

    # Create the plot with a grid of subplots (one for each feature pair)
    fig, ax = plt.subplots(num_feat, num_feat, figsize=(12, 12))
    plt.set_cmap('viridis')  # set the color map for better visualization


    # make the visualisation
    import matplotlib as mpl
    hist_col = plt.get_cmap('viridis', K).colors

    # Loop over each pair of features
    for feature1 in range(num_feat):
        # Set the label for the axis
        ax[feature1, 0].set_ylabel(feature_names[feature1])
        ax[0, feature1].set_xlabel(feature_names[feature1])
        ax[0, feature1].xaxis.set_label_position('top')

        for feature2 in range(num_feat):
            # Extract the data for the feature pair
            x_data = data[:, feature1]
            y_data = data[:, feature2]

            # Create scatter plot for off-diagonal elements
            if feature1 != feature2:
                ax[feature1, feature2].scatter(x_data, y_data, c=cluster_ids)
            else:
                # Sort the labels and data so that classes are in order
                inds = np.argsort(cluster_ids)
                sorted_y = cluster_ids[inds]
                sorted_x = x_data[inds]

                # Split the data into different classes
                splits = np.split(sorted_x, np.unique(sorted_y, return_index=True)[1][1:])

                # Plot the histogram
                for i, split in enumerate(splits):
                    ax[feature1, feature2].hist(split, bins=20, color=hist_col[i], edgecolor='black')

    # remember to put your user name into the title as specified
    username = "d3-yadav (Dikshit Yadav_24030152)"  # Replace with your UWE username
    fig.suptitle(f'Visualisation of {K} clusters by {username}', fontsize=16, y=0.925)


    # save it to file as specified
    fig.savefig('myVisualisation.jpg')



    return fig,ax

    # <==== insert your code above here
