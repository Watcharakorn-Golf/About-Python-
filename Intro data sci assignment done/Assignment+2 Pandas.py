
# coding: utf-8

# ----
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Assignment 2 - Pandas Introduction
# All questions are weighted the same in this assignment.
# ## Part 1
# The following code loads the olympics dataset (olympics.csv), which was derrived from the Wikipedia entry on [All Time Olympic Games Medals](https://en.wikipedia.org/wiki/All-time_Olympic_Games_medal_table), and does some basic data cleaning. 
# 
# The columns are organized as # of Summer games, Summer medals, # of Winter games, Winter medals, total # number of games, total # of medals. Use this dataset to answer the questions below.

# In[1]:


import pandas as pd

df = pd.read_csv('olympics.csv', index_col=0, skiprows=1)

for col in df.columns:
    if col[:2]=='01':
        df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
    if col[:2]=='02':
        df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
    if col[:2]=='03':
        df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
    if col[:1]=='№':
        df.rename(columns={col:'#'+col[1:]}, inplace=True)

names_ids = df.index.str.split('\s\(') # split the index by '('

df.index = names_ids.str[0] # the [0] element is the country name (new index) 
df['ID'] = names_ids.str[1].str[:3] # the [1] element is the abbreviation or ID (take first 3 characters from that)

df = df.drop('Totals')
df.head()


# ### Question 0 (Example)
# 
# What is the first country in df?
# 
# *This function should return a Series.*

# In[2]:


# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the row for Afghanistan, which is a Series object. The assignment
    # question description will tell you the general format the autograder is expecting
    return df.iloc[0]

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
answer_zero() 


# ### Question 1
# Which country has won the most gold medals in summer games?
# 
# *This function should return a single string value.*

# In[3]:


def answer_one():
    summer = df['Gold']
    summer = summer.reset_index()
    cty = ''
    most = 0
    n= 0
    #print(summer)
    for new_most in summer.values:
        #print(new_most)
        if new_most[1] > most:
            most = new_most[1]
           # print(new_most)
            cty =   new_most[0]  
            n += 1
    ans_one = cty   
    return ans_one

answer_one()


# ### Question 2
# Which country had the biggest difference between their summer and winter gold medal counts?
# 
# *This function should return a single string value.*

# In[4]:


def answer_two():
    summer = df['Gold']
    winter = df['Gold.1']
    biggest = 0
    for c in range(len(df)):
        diff = abs(summer[c] - winter[c])
        #print(diff)
        if diff >= biggest:
            biggest = diff
            cty = df.where(summer == summer[c])
            
    #print("Biggest defference is",biggest)      
    ans_two = cty.loc[:,['Gold','Gold.1']].dropna()
    #ans_two['Difference'] = biggest
    ans_two = ans_two.index[0]
    return ans_two
answer_two()


# ### Question 3
# Which country has the biggest difference between their summer gold medal counts and winter gold medal counts relative to their total gold medal count? 
# 
# $$\frac{Summer~Gold - Winter~Gold}{Total~Gold}$$
# 
# Only include countries that have won at least 1 gold in both summer and winter.
# 
# *This function should return a single string value.*

# In[5]:


def answer_three():
    SumAndWin = df[(df['Gold'] > 0) & (df['Gold.1'] > 0)]
    summer = SumAndWin['Gold']
    winter = SumAndWin['Gold.1']
    total_games = SumAndWin['Gold.2']
    biggest_relativity = 0
    Total = 0
    for c in range(len(SumAndWin)):
        diff = abs(summer[c] - winter[c])
        total = total_games[c]
        diff_relativity = diff / total
        if diff_relativity >= biggest_relativity:
            biggest_relativity = diff_relativity
            cty = df.where(summer == summer[c])    
            Total = total
            #print('Gold Medal',summer[c],winter[c])
            #print("Gold Total {}+{}+{} ={}".format(total_sum[c],total_win[c],total_games[c],total))
    ans_three = cty.loc[:,['Gold','Gold.1']].dropna()
    ans_three['Total Gold'] = Total
    ans_three['Difference'] = biggest_relativity
    ans = ans_three.index[0]
    return ans
answer_three()


# ### Question 4
# Write a function that creates a Series called "Points" which is a weighted value where each gold medal (`Gold.2`) counts for 3 points, silver medals (`Silver.2`) for 2 points, and bronze medals (`Bronze.2`) for 1 point. The function should return only the column (a Series object) which you created, with the country names as indices.
# 
# *This function should return a Series named `Points` of length 146*

# In[6]:


def answer_four():
    cl = df.columns[-5:-2]
    #print(cl)
    games = df[cl]
    #print(games.head())
    Gold = games['Gold.2']
    Silver = games['Silver.2']
    Bronze = games['Bronze.2']
    Points = pd.Series()
    for n in range(len(games)):
        point = (Gold[n]*3) + (Silver[n]*2) + (Bronze[n]*1)
        Points.loc[games.index[n]] = point
    
    #df['Points'] = pd.Series((df['Gold.2']*3) + (df['Silver.2']*2) + df['Bronze.2'])  
    return Points
answer_four()


# ## Part 2
# For the next set of questions, we will be using census data from the [United States Census Bureau](http://www.census.gov). Counties are political and geographic subdivisions of states in the United States. This dataset contains population data for counties and states in the US from 2010 to 2015. [See this document](https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2010-2015/co-est2015-alldata.pdf) for a description of the variable names.
# 
# The census dataset (census.csv) should be loaded as census_df. Answer questions using this as appropriate.
# 
# ### Question 5
# Which state has the most counties in it? (hint: consider the sumlevel key carefully! You'll need this for future questions too...)
# 
# *This function should return a single string value.*

# In[7]:


census_df = pd.read_csv('census.csv')
census_df.head()


# In[32]:


def answer_five():
    keep_col = ['SUMLEV','STNAME','CTYNAME']
    census = census_df.set_index(['STATE','SUMLEV'])
    test = census_df.values
    #print(test)
    i_total = 0
    i = 0
    state_cnt = pd.Series()
    for n in range(len(census)):
        x = census.index[n]
        #print(x)
        STATE = x[0] 
        if x[1] == 40 :
            i = 0
            i_total += 1
            #print(x[0],x[1])
            #print('Change the State to', STATE,i)
        else:
            i_total += 1
            i += 1
            #print(x[0],x[1],i) 
            state_cnt.loc[STATE] = i
            #print(STATE)      
    state_cnt.loc[STATE] = i  
    MOST_STNAME = []   
    MOST_STNAME = state_cnt.sort_values()
    state = MOST_STNAME.index[-1]
    numofcnty = MOST_STNAME.iloc[-1]
    col = ['COUNTY','STNAME']
    cencuss = census_df[col]
    test = cencuss.iloc[state]
    return census_df.groupby(['STNAME']).sum()['COUNTY'].idxmax()
answer_five()


# ### Question 6
# **Only looking at the three most populous counties for each state**, what are the three most populous states (in order of highest population to lowest population)? Use `CENSUS2010POP`.
# 
# *This function should return a list of string values.*

# In[42]:


def answer_six():
    pop = census_df.set_index(['STNAME','CENSUS2010POP','CTYNAME'])
    pop = pop.sort_index(ascending= False) 
    pops = pop.reset_index()
    #pop = pop.set_index(['STNAME','CENSUS2010POP'])
    ans = []
    test = ''
    most_pop_cnty = ''
    i = 0
    n = 0
    for st in pops.values:
        words = st[0:3]
        i += 1            
        if (st[0] == st[2]):
            i = 1        
        if (i > 1) and (i <= 6):
                #most_pop_cnty = most_pop_cnty + (str(words[0]) + ' ' + str(words[2]))
                most_pop_cnty = (str(words[0]) + ' ' + str(words[2]))
                test = test + ' ' + most_pop_cnty
                i += 1
                n += 1
                #print(test)
                if n == 3 :
                    ans.append(test)
                    #print(test)
                    test = ''
                    n = 1
    return census_df[census_df['SUMLEV']==50].groupby('STNAME')['CENSUS2010POP'].apply(lambda x: x.nlargest(3).sum()).nlargest(3).index.values.tolist()

answer_six()


# ### Question 7
# Which county has had the largest absolute change in population within the period 2010-2015? (Hint: population values are stored in columns POPESTIMATE2010 through POPESTIMATE2015, you need to consider all six columns.)
# 
# e.g. If County Population in the 5 year period is 100, 120, 80, 105, 100, 130, then its largest change in the period would be |130-80| = 50.
# 
# *This function should return a single string value.*

# In[10]:


def answer_seven():
    #popes = census_df.set_index['POPESTIMATE2010' ,'POPESTIMATE2011'] 
    popes = census_df.set_index(['CTYNAME','POPESTIMATE2010','POPESTIMATE2011','POPESTIMATE2012',
                                 'POPESTIMATE2013','POPESTIMATE2014','POPESTIMATE2015'])
    largest_change = pd.Series()
    for es in popes.index:
        big_num = 0
        big_diff = 0
        #print(es)
        cnt_name = es[0]
        pop_period = es[1:]
        #print(cnt_name, pop_period)
        for pop_num_two in pop_period:
            
            for pop_num in pop_period:
                diff = abs(pop_num - pop_num_two)
                #print(pop_num,pop_num_two,diff) 
                if diff >= big_diff:
                    big_diff = diff
                    big_num = pop_num
            #print(pop_num,pop_num_two,diff)       
        #print(big_diff)
        largest_change.loc[cnt_name] = big_diff
        sort_values = largest_change.sort_values(ascending=False)
        ans_seven = str(sort_values.index[0])+ " " + str(sort_values[0])
        ans = sort_values.index[0]
        test = str(sort_values[0])
        county_only = census_df[census_df['SUMLEV']==50].set_index('CTYNAME')
        years = ['POPESTIMATE2010','POPESTIMATE2011','POPESTIMATE2012','POPESTIMATE2013','POPESTIMATE2014','POPESTIMATE2015']
    return (county_only.loc[:, years].max(axis=1) - county_only.loc[:, years].min(axis=1)).argmax()
answer_seven()


# ### Question 8
# In this datafile, the United States is broken up into four regions using the "REGION" column. 
# 
# Create a query that finds the counties that belong to regions 1 or 2, whose name starts with 'Washington', and whose POPESTIMATE2015 was greater than their POPESTIMATE 2014.
# 
# *This function should return a 5x2 DataFrame with the columns = ['STNAME', 'CTYNAME'] and the same index ID as the census_df (sorted ascending by index).*

# In[11]:


def answer_eight():
    keep_columns = ['SUMLEV','REGION','STNAME', 'CTYNAME','POPESTIMATE2014','POPESTIMATE2015']
    two_col = ['STNAME', 'CTYNAME']
    census = census_df[keep_columns]
    rel = census[(census_df['SUMLEV']==50) & ((census['REGION'] == 1) | (census['REGION'] == 2))
                 & (census['POPESTIMATE2015'] > census['POPESTIMATE2014'])]
    rel = rel.reset_index()
    two_data = rel[two_col]
    Threshold_word = 'Washington'
    n = 0
    for name in two_data.values:
        wordsplit = name[1].split()
        #print(name)
        #print(wordsplit[0])

        if (name[0] != name[1]) and ( name[0] != Threshold_word) and (wordsplit[0] == Threshold_word):
            two_data['CTYNAME'].loc[n] =  name[1]
        n += 1    
    eiei = census_df[two_col]
    ans = two_data[ (two_data['CTYNAME'] == 'Washington County') ]
    #ans = two_data['CTYNAME']
    #ans = ans.reset_index()
    ans_eight = ans[two_col]
    print(ans_eight)
    region_county = census_df[(census_df['SUMLEV']==50) & ((census_df['REGION'] ==1) | (census_df['REGION'] ==2))]
    start_washington = region_county[region_county['CTYNAME'].str.startswith('Washington')]
    popestimate_comparison = start_washington[start_washington['POPESTIMATE2015']> start_washington['POPESTIMATE2014']]
    return popestimate_comparison[['STNAME','CTYNAME']]

answer_eight()


# In[ ]:




