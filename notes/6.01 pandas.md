
# Data analysis with `pandas`

Basic concepts:

- a table with multiple columns is a `DataFrame`
- a single column on its own is a `Series`

Basic `pandas` commands for analyzing data. Good for use in iPython notebooks.

Note: columns here are ambiguous in their datatypes; these are just illustrations.

    import pandas as pd
    import pylab as plt

    # Load data from csv into a DataFrame
    df = pd.read_csv('data.csv')

    # Get top 5 rows
    df.head()

    # Get top 10 rows
    df.head(10)

    # Get bottom 5 rows
    df.tail()

    # Get bottom 10 rows
    df.tail(10)

    # See overview of data
    # (columns, number non-null values, datatypes)
    df.info()

    # See number of rows
    len(df)

    # See columns
    df.columns

    # Select a column (as a Series)
    df['col_1']

    # Operate on rows of the column
    df['col_1'] + 10
    df['col_1'] * 10
    df['col_1'] + df['col_2']
    # etc

    # Round values of a column
    df['col_1'].round()

    # Create a new column
    df['col_3'] = df['col_1'] + 10

    # Sort by a column
    df.sort('col_1')

    # Set an index (the column you use becomes the index)
    # This overwrites the existing index
    df.set_index('col_1')

    # Set an index w/o overwriting the existing index (append)
    df.set_index('col_1', append=True)

    # Set multiple indices
    df.set_index(['col_1', 'col_2'])

    # For speed benefits, you should sort your index
    df.sort_index()

    # Access item(s) by index value
    df.loc['some value']

    # Remove an index/indices
    df.reset_index('col_1')
    df.reset_index(['col_1', 'col_2'])

    # Select multiple columns
    df[['col_1', 'col_2']]

    # See datatypes
    df.dtypes

    # See some descriptive statistics
    # (e.g. count, mean, std, etc)
    df.describe()

    # Descriptive statistics for a single column
    df['col_1'].describe()

    # Get a particular descriptive statistics
    df['col_1'].mean()
    df['col_1'].median()
    df['col_1'].mode()
    df['col_1'].std()
    df['col_1'].max()
    df['col_1'].min()

    # See unique values for a column
    df['col_1'].unique()

    # Count num of occurences for values in a column
    df['col_1'].value_counts()

    # Generate cross tab of two columns
    pd.crosstab(df['col_1'], df['col_2'])

    # Convert a string representation to a number representation
    str_vals = df['col_1'].unique()
    mapping = dict(zip(str_vals, range(0, len(str_vals) + 1)))
    df['col_1_as_int'] = df['col_1'].map(mapping).astype(int)

    # Select rows where a value satisfies a condition
    df[df['col_1'] == True]

    # Select rows where values satisfy multiple conditions
    df[(df['col_1'] == True) & (df['col_2'] == True)] # and
    df[(df['col_1'] == True) | (df['col_2'] == True)] # or

    # Count rows
    df[df['col_1'] == True].count()

    # Get rows where columns have non-null values
    df[df['col_1'].notnull()]

    # Get rows where a column has a null value
    df[df['col_1'].isnull()]

    # Drop rows with any null values
    df.dropna(axis=0, how='any')

    # Drop rows of all null values
    df.dropna(axis=0, how='all')

    # Drop columns with null values
    df.dropna(axis=1)

    # Replace column's nan values with average for the column (in place)
    df.replace({
        'col_1': {nan: df['col_1'].mean()}
    }, inplace=True)

    # Or more easily, replace nan values
    df['col_1'].fillna(df['col_1'].mean(), inplace=True)

    # Use "forward fill" (take previous non-nan values
    # and and fill downward until next non-nan value)
    df['col_1'].fillna(method='ffill')

    # Use "backward fill"
    df['col_1'].fillna(method='bfill')

    # Combine two dataframes (i.e. add columns)
    pd.concat([df, df2], axis=1)

    # Stack two dataframes (i.e. add rows)
    pd.concat([df, df2], axis=0)

    # Group by column
    df.groupby('col_1')

    # Group by columns
    df.groupby(['col_1', 'col_2'])

    # Count number of rows in groups
    # (can also do all the other descriptive statistics)
    df.groupby('col_1').size()

    # For example, get max of each group:
    df.groupby('col_1').max()

    # Get mean of each group:
    df.groupby('col_1').mean()
    # etc

    # Use multiple descriptors
    df.groupby('col_1').agg(['min', 'max'])

    # You can also group by manipulations of columns
    # For example, group by decade
    df.groupby(df['year'] // 10 * 10)

    # Drop a column
    df.drop('col_1', axis=1)

    # Drop multiple columns
    df.drop(['col_1', 'col_2'], axis=)

    # Get data as numpy array
    df.values

    # Find rows that satisfy a string condition
    df['col_1'].str.startswith('foo')
    df['col_1'].str.endswith('foo')
    df['col_1'].str.contains('foo')
    df['col_1'].str.len() > 10
    df['col_1'].str.slice(0, 5)
    # etc

    # Turn rows or indices into columns
    # (useful when doing groupbys with multiple columns)
    df.unstack('col_1')

    # Turn columns into indices
    df.stack()

    # Merge dataframes (automatically looks for matching columns)
    df.merge(another_df)

    # You can be explicit about matching columns
    df.merge(another_df, on=['col_1', 'col_2'])




## Dealing with datetimes

    # Turn years into decades
    df['year'] // 10 * 10

    # Datetime slice (requires that your DataFrame has a DateTimeIndex)
    start = datetime(year=1980, month=1, day=1, hour=0, minute=0)
    end = datetime(year=1981, month=1, day=1, hour=0, minute=0)
    df[start:end]

    # Group by some time interval,
    # assuming DateTimeIndex
    df.groupby(pd.TimeGrouper(freq='M')) # month
    df.groupby(pd.TimeGrouper(freq='10Min')) # 10 minutes
    # etc

    # Access different datetime resolutions
    df['date_col'].dt.year
    df['date_col'].dt.month
    df['date_col'].dt.day
    df['date_col'].dt.dayofyear
    df['date_col'].dt.dayofweek
    df['date_col'].dt.weekday
    # etc

    # For DateTime indices, you can do something similar
    df.index.weekday
    # etc

    # Combine rows by some interval
    df.resample('M', how='mean') # month
    df.resample('M', how=np.median) # month
    # etc


## Loading data

    # Load from CSV
    df = pd.read_csv('data.csv')

    # If you have an index column in the file, you can do:
    df = pd.read_csv('data.csv', index_col='col1')

    # Directly parse dates
    df = pd.read_csv('data.csv', parse_dates=True)
    df = pd.read_csv('data.csv', parse_dates=['date'])

    # Skip rows
    df = pd.read_csv('data.csv', skiprows=10)


## Plotting

### Initial setup

    import matplotlib.pyplot as plt

    # Set the global default size of matplotlib figures
    plt.rc('figure', figsize=(10, 5))


### Basics

    # Plot a Series (x=index, y=values)
    sr.plot()

    # Plot a Dataframe
    df.plot(x='col_1', y='col_2')

    # Specify figsize
    df.plot(figsize=(15,10))

    # Specify y limits (similar for x limits)
    df.plot(ylim=[0, 100])

    # Plot multiple columns against the index
    df.plot(subplots=True)


### Plot a cross tab

    # Generate cross tab of two columns
    xt = pd.crosstab(df['col_1'], df['col_2'])

    # Normalize
    xt_norm = xt.div(xt.sum(1).astype(float), axis=0)

    # Plot as stacked bar chart
    xt_norm.plot(kind='bar',
                 stacked=True,
                 title='cross tab plot')
    plt.xlabel('col_1')
    plt.ylabel('col_2')


### Plot subplots as a grid

    # Plot a grid of subplots
    fig = plt.figure(figsize=(10, 10))
    fig_dims = (3,2)

    plt.subplot2grid(fig_dims, (0, 0))
    df['col_1'].value_counts().plot(kind='bar', title='column 1')

    plt.subplot2grid(fig_dims, (0, 1))
    df['col_2'].hist()
    plt.title('histogram of column 2')


### Plot overlays

    for i in range(3):
        df[df['col_1'] == i].plot(kind='kde', alpha=0.5)
    plt.title('overlaid density plots')
    plt.legend(('class 1', 'class 2', 'class 3'), loc='best')


### Other plots

    # Scatter plot
    plt.scatter(df['col_1'], df['col_2'])
    plt.title('scatter plot')
    plt.xlabel('column 1')
    plt.ylabel('column 2')

    # Histogram
    bin_size = 10
    max_val = df['col_1'].max()
    df['col_1'].hist(bins=max_val/bin_size, range=(1, max_val))

    # Stacked histogram
    df1 = df[df['col_1'] == True]['col_2']
    df2 = df[df['col_1'] == False]['col_2']
    plt.hist([df1, df2], bins=max_val/bin_size, range=(1, max_val))
    plt.legend(('true col_2', 'false col_2'), loc='best')

### Decorating plots

    plt.title('some title')
    plt.xlabel('column 1')
    plt.ylabel('column 2')
    plt.grid(True)
    plt.legend(['col 1', 'col 2'], loc='best')

### Saving a figure

    plt.savefig('filename.png')


## iPython Notebooks

Run the iPython notebook server:

    $ ipython notebook --pylab inline

The `--pylab inline` option renders plots in the notebook.

Alternatively, you can add:

    %matplotlib inline

To the first cell in the notebook.

iPython notebooks have a command and edit (insert) mode.

Hit `esc` to enter command mode, and hit `h` to see the available commands.

Some useful commands:

- `A`: insert cell above
- `B`: insert cell below
- `shift+enter`: execute current cell and go to next cell (also works in edit mode)
- `ctrl+enter`: execute current cell (also works in edit mode)
