import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="whitegrid")
from scipy.stats import pearsonr

class Correlation:

    def __init__(self, df):
        self.df = df


    def get_pairs(self):
        """ Get diagonal and lower triangular pairs of correlation matrix. Unique combinations.
        Will use these pairs to drop duplicates.
        """
        pairs = set()
        cols = self.df.columns
        
        for i in range(0, self.df.shape[1]):
            for j in range(0, i+1):
                pairs.add((cols[i], cols[j]))
        
        return pairs


    def get_correlation_tall(self):
        """ Correlation of each unique combination of columns in self.df
        Note: Combinations not permutations 
        """
        r2 = self.df.corr().unstack()
        labels = self.get_pairs()
        r2 = r2.drop(labels=labels).to_frame(name = 'Correlation').reset_index()
        return r2


    def get_correlation_wide(self):
        return self.df.corr()


    def get_pearson_p_values(self):
        """ Calculate P-value associated with Pearson Correlation
        """
        dfcols = pd.DataFrame(columns=self.df.columns)
        pvalues = dfcols.transpose().join(dfcols, how='outer')

        for r in self.df.columns:
            for c in self.df.columns:
                tmp = self.df[self.df[r].notnull() & self.df[c].notnull()]
                pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 6)
        
        labels = self.get_pairs()
        pvalues = pvalues.unstack().drop(labels=labels).to_frame(name = 'p-value').reset_index()

        return pvalues
    

    def get_correlation_summary(self):
        x = self.get_correlation_tall()
        y = self.get_pearson_p_values()

        return x.merge(y, on = ['level_0', 'level_1'], how = 'outer')


    def plot_corr(self, vmin = -1, vmax = 1, center = 0, title = 'Correlation'):

        def corr_sig(df=None):
            p_matrix = np.zeros(shape=(self.df.shape[1],df.shape[1]))
            for col in df.columns:
                for col2 in df.drop(col,axis=1).columns:
                    _ , p = stats.pearsonr(df[col],df[col2])
                    p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
            return p_matrix

        p_values = corr_sig(self.df)
        mask = np.invert(np.tril(p_values<0.05)) 

        f, ax = plt.subplots(figsize=(10, 5))

        cmap = sns.diverging_palette(20, 230, as_cmap=True)

        corr = self.df.corr().round(2)
        
        ax.set_title(title)

        sns.heatmap(corr, mask=mask | np.triu(np.ones_like(corr, dtype=bool)), cmap=cmap,  center=center, vmin = vmin, vmax = vmax,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={'fontsize':7}, annot = True)
        
        # plt.savefig(os.path.join(img_dir, 'Correlation.png'), bbox_inches = 'tight')
        
        plt.show()

  