# This is a file for junk code. It occasionally comes in handy in case I change my mind about using the code.

# def get_cc_means(data, columns=COLS_FOR_SUMMARY):
#     cc_means = {}
#     yes_means = data[data.Graduated == 'Yes'].mean() # returns class conditional means for all relevant columns
#     no_means =  data[data.Graduated == 'No'].mean()
#     for col in columns: 
#         cc_means[col] = {'Yes': yes_means[col], 'No': no_means[col]}
#         # cc_means[col][y/n] gives class-conditional mean
#     return cc_means 


# for i in range(len(data.index)):
#     for col in cc_means:
#         if data.ix[i][col].isnull():
#             data.ix[i][col] = ccmeans[col][data.ix[i]['Graduated']]

# data.apply(lambda x: means[col.Graduated] if pd.isnull(x[col]) else x[col])

    # for col in columns:
    #     for i in set(data[groupvar]):
    #         data[col].loc[(data[col].isnull()) & (data[groupvar] == i)] = means[col][i]

        # for i in set(data[groupvar]):
        #     data[col].loc[(data[col].isnull()) & (data[groupvar] == i)] = means[col][i]
