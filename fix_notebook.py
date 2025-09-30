import nbformat
from nbformat.v4 import new_code_cell
import re

def create_vif_function_cell():
    source = '''# Reusable VIF calculation function
def calculate_vif(dataframe):
    """
    Calculate Variance Inflation Factor for all features in a dataframe.
    
    Parameters:
    dataframe: pandas DataFrame with numeric features
    
    Returns:
    DataFrame with Features and their VIF values, sorted by VIF descending
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif = pd.DataFrame()
    vif['Features'] = dataframe.columns
    vif['VIF'] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by="VIF", ascending=False)
    return vif'''
    
    return new_code_cell(source)

def fix_notebook(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    vif_function_added = False
    cells_to_modify = []
    
    for i, cell in enumerate(nb.cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            modified = False
            
            if "num_vars = ['mnth', 'weekday', 'temp', 'hum', 'windspeed','casual','registered','cnt']" in source:
                source = source.replace(
                    "num_vars = ['mnth', 'weekday', 'temp', 'hum', 'windspeed','casual','registered','cnt']",
                    "num_vars = ['mnth', 'weekday', 'temp', 'hum', 'windspeed','cnt']"
                )
                modified = True
            
            if "sns.pairplot(dataset[['temp','hum','windspeed','casual','registered','cnt']])" in source:
                source = source.replace(
                    "sns.pairplot(dataset[['temp','hum','windspeed','casual','registered','cnt']])",
                    "sns.pairplot(dataset[['temp','hum','windspeed','cnt']])"
                )
                modified = True
            
            if "sns.pairplot(df_train[['temp','casual','registered','cnt']])" in source:
                source = source.replace(
                    "sns.pairplot(df_train[['temp','casual','registered','cnt']])",
                    "sns.pairplot(df_train[['temp','cnt']])"
                )
                modified = True
            
            if "X_train_lm = sm.add_constant(X_train[['registered']])" in source:
                source = source.replace(
                    "X_train_lm = sm.add_constant(X_train[['registered']])",
                    "X_train_lm = sm.add_constant(X_train[['temp']])"
                )
                modified = True
            
            if "X_train_lm = X_train[['registered', 'casual','temp']]" in source:
                source = source.replace(
                    "X_train_lm = X_train[['registered', 'casual','temp']]",
                    "X_train_lm = X_train[['temp', 'yr', 'season']]"
                )
                modified = True
            
            if "plt.plot(X_train_lm.iloc[:, 1], 0.0245 + 0.933*X_train_lm.iloc[:, 1], 'r')" in source:
                source = source.replace(
                    "plt.plot(X_train_lm.iloc[:, 1], 0.0245 + 0.933*X_train_lm.iloc[:, 1], 'r')",
                    "# Regression line coefficients will vary based on actual data"
                )
                modified = True
            
            if not vif_function_added and 'variance_inflation_factor' in source and 'from statsmodels.stats.outliers_influence import variance_inflation_factor' in source:
                cells_to_modify.append((i, 'insert_vif_function'))
                vif_function_added = True
            
            if 'vif = pd.DataFrame()' in source and 'vif[\'Features\'] = X' in source:
                if vif_function_added:
                    dataframe_name = None
                    if "vif['Features'] = X_train.columns" in source:
                        dataframe_name = 'X_train'
                    elif "vif['Features'] = X.columns" in source:
                        dataframe_name = 'X'
                    
                    if dataframe_name:
                        source = f"vif = calculate_vif({dataframe_name})\nvif"
                        modified = True
            
            if modified:
                cell['source'] = source
    
    for idx, action in reversed(cells_to_modify):
        if action == 'insert_vif_function':
            nb.cells.insert(idx, create_vif_function_cell())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Notebook fixed and saved to {output_path}")
    print(f"VIF function added: {vif_function_added}")

if __name__ == '__main__':
    fix_notebook('Bike Sharing Case Study.ipynb', 'Bike Sharing Case Study.ipynb')
