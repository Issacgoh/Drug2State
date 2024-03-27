#!/usr/bin/env python
# coding: utf-8

# 
# ## Download data base

# In[1]:


get_ipython().system(' wget -O /nfs/team205/ig7/projects/SCC_nano_string/drug2state/resources/chembl_33_sqlite.tar.gz https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_33/chembl_33_sqlite.tar.gz')


# In[2]:


# at farm
# download checksums file
get_ipython().system(' wget -O /nfs/team205/ig7/projects/SCC_nano_string/drug2state/resources/db33_chechsums.txt https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_33/checksums.txt')


# In[3]:


get_ipython().system(' cat /nfs/team205/ig7/projects/SCC_nano_string/drug2state/resources/db33_chechsums.txt')


# In[4]:


get_ipython().system('sha256sum /nfs/team205/ig7/projects/SCC_nano_string/drug2state/resources/chembl_33_sqlite.tar.gz')


# In[5]:


# decompress
get_ipython().system(' tar -xzvf /nfs/team205/ig7/projects/SCC_nano_string/drug2state/resources/chembl_33_sqlite.tar.gz')


# # Add modules:
#         - BAO (mol mech)
#         - Add mol description
#         - Add GO

# ## Import required modules

# In[1]:


import pandas as pd
pd.set_option('display.max_rows', 600)
import numpy as np
import sqlite3


# In[2]:


conn = sqlite3.connect('/nfs/team205/ig7/projects/SCC_nano_string/drug2state/resources/chembl_33/chembl_33_sqlite/chembl_33.db')
# Create a cursor object using the cursor() method
cursor = conn.cursor()
# Retrieve the list of all tables in the database
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# Fetch all results from the cursor into a list
tables = cursor.fetchall()
con = conn
tables


# In[3]:


db_x_ont = pd.read_sql_query("SELECT * from BIOASSAY_ONTOLOGY", con)
print(db_x_ont.shape)
prefix = "db_x_ont"
db_x_ont.columns = [prefix +'|' + x for x in db_x_ont.columns]

# merge using Assays: BAO_FORMAT
db_x_ont.head()


# # Molecular structure

# In[4]:


db_x_smiles = pd.read_sql_query("SELECT * from compound_structures", con)
print(db_x_smiles.shape)
prefix = "db_x_smiles"
db_x_smiles.columns = [prefix +'|' + x for x in db_x_smiles.columns]
db_x_smiles = db_x_smiles[['db_x_smiles|molregno','db_x_smiles|canonical_smiles']]
db_x_smiles.head()


# ## Activity data loading

# In[5]:


get_ipython().run_cell_magic('time', '', 'activities = pd.read_sql_query("SELECT * from activities", con)\nprint(activities.shape)\n\n#\xa0rename columns to be able to track back\nactivities.columns=[f\'activities|{x}\' for x in activities.columns]\n\nprint(activities.shape)\nactivities.head()')


# # Go terms loading
# - component_synonyms -> Component_go:GO_ID -> GO_CLASSIFICATION:PREF_NAME

# In[6]:


db_x_COMPONENT_GO = pd.read_sql_query("SELECT * from COMPONENT_GO", con)
print(db_x_COMPONENT_GO.shape)
prefix = "db_x_COMPONENT_GO"
db_x_COMPONENT_GO.columns = [prefix +'|' + x for x in db_x_COMPONENT_GO.columns]

# merge using Assays: BAO_FORMAT
db_x_COMPONENT_GO = db_x_COMPONENT_GO[[prefix+'|component_id',prefix+'|go_id']]
db_x_COMPONENT_GO.head()


# In[7]:


db_x_GO_CLASSIFICATION = pd.read_sql_query("SELECT * from GO_CLASSIFICATION", con)
print(db_x_GO_CLASSIFICATION.shape)
prefix = "db_x_GO_CLASSIFICATION"
db_x_GO_CLASSIFICATION.columns = [prefix +'|' + x for x in db_x_GO_CLASSIFICATION.columns]

# merge using Assays: BAO_FORMAT
db_x_GO_CLASSIFICATION = db_x_GO_CLASSIFICATION[[prefix+'|go_id',prefix+'|path']]#prefix+'|parent_go_id'
db_x_GO_CLASSIFICATION.head()


# ## Assay data loading

# In[8]:


# assay data
assays = pd.read_sql_query("SELECT * from assays", con)
print(assays.shape)

# rename columns to be able to track back
assays.columns=[f'assays|{x}' for x in assays.columns]

print(assays.shape)
assays.head()


# In[9]:


assays.columns


# ## Target data loading

# In[10]:


target_dictionary = pd.read_sql_query("SELECT * from target_dictionary", con)
print(target_dictionary.shape)

# rename columns to be able to track back
target_dictionary.columns=[f'target_dictionary|{x}' for x in target_dictionary.columns]

target_dictionary.head()


# In[11]:


target_components = pd.read_sql_query("SELECT * from target_components", con)
print(target_components.shape)

# rename columns to be able to track back
target_components.columns=[f'target_components|{x}' for x in target_components.columns]

target_components.head()


# In[12]:


component_synonyms = pd.read_sql_query("SELECT * from component_synonyms", con)
print(component_synonyms.shape)

# rename columns to be able to track back
component_synonyms.columns=[f'component_synonyms|{x}' for x in component_synonyms.columns]

component_synonyms.head()


# ## Compound data loading

# In[13]:


drug_mechanism = pd.read_sql_query("SELECT * from drug_mechanism", con)
print(drug_mechanism.shape)

# rename columns to be able to track back
drug_mechanism.columns=[f'drug_mechanism|{x}' for x in drug_mechanism.columns]

drug_mechanism.head()


# In[14]:


drug_mechanism


# In[15]:


molecule_dictionary = pd.read_sql_query("SELECT * from molecule_dictionary", con)
print(molecule_dictionary.shape)

# rename columns to be able to track back
molecule_dictionary.columns=[f'molecule_dictionary|{x}' for x in molecule_dictionary.columns]

molecule_dictionary.head()


# In[16]:


molecule_atc_classification = pd.read_sql_query("SELECT * from molecule_atc_classification", con)
print(molecule_atc_classification.shape)

# rename columns to be able to track back
molecule_atc_classification.columns=[f'molecule_atc_classification|{x}' for x in molecule_atc_classification.columns]

molecule_atc_classification.head()


# In[17]:


atc_classification = pd.read_sql_query("SELECT * from atc_classification", con)
print(atc_classification.shape)

# rename columns to be able to track back
atc_classification.columns=[f'atc_classification|{x}' for x in atc_classification.columns]

atc_classification.head()


# In[18]:


atc_classification


# In[19]:


# atc_classification = pd.read_sql_query("SELECT * from atc_classification", con)
# atc_classification.to_csv('/home/jovyan/projects/P50_ChEMBL/csv/atc_classification_db30.csv',index=False)


# ## Concatenate information

# * activities_final:<br> 
# 'activities|activity_id', 'activities|assay_id', 'activities|molregno',
# 'activities|pchembl_value', 'activities|type', 'activities|standard_relation', 'activities|standard_value',
# 'activities|standard_units', 'activities|standard_flag',
# 'activities|standard_type', 'activities|activity_comment',
# 'assays|description','assays|assay_type','assays|tid', 'assays|confidence_score','assays|curated_by','assays|chembl_id',<br>
# <br>
# * targets_final:<br>
# 'target_dictionary|tid','target_dictionary|target_type','target_dictionary|pref_name','target_dictionary|organism','target_dictionary|chembl_id',
# 'component_synonyms|component_synonym', 'component_synonyms|syn_type'<br>
# <br>
# * molecule_dictionary:<br>
# 'molecule_dictionary|molregno', 'molecule_dictionary|pref_name','molecule_dictionary|chembl_id', 'molecule_dictionary|max_phase', 'molecule_dictionary|molecule_type','molecule_dictionary|oral',
# 'molecule_dictionary|parenteral', 'molecule_dictionary|topical','molecule_dictionary|black_box_warning','molecule_dictionary|natural_product'<br>
# <br>    
# * drug_mechanism:<br>
# 'drug_mechanism|molregno','drug_mechanism|mechanism_of_action','drug_mechanism|tid','drug_mechanism|action_type',
# <br>
# * molecule_atc_classification:<br>
# 'molecule_atc_classification|mol_atc_id','molecule_atc_classification|level5','molecule_atc_classification|molregno'
# <br>
# * atc_classification:<br>
# 'atc_classification|who_name', 'atc_classification|level1','atc_classification|level2', 'atc_classification|level3',
# 'atc_classification|level4', 'atc_classification|level5','atc_classification|level1_description','atc_classification|level2_description',
# 'atc_classification|level3_description','atc_classification|level4_description'

# In[20]:


# merge activities and assays data
final_df = activities.merge(assays,how='left',left_on='activities|assay_id',right_on='assays|assay_id')
final_df = final_df[['activities|activity_id', 'activities|assay_id', 'activities|molregno',
                             'activities|pchembl_value', 'activities|type', 'activities|standard_relation', 'activities|standard_value',
                             'activities|standard_units', 'activities|standard_flag','activities|standard_type', 'activities|activity_comment',
                             'assays|description','assays|assay_type','assays|tid', 'assays|confidence_score','assays|curated_by','assays|chembl_id','assays|bao_format']]
print(f'activities+assays: {final_df.shape}')


# merge activities and drug_mechanism based on 'molregno': how='outer' to capture all
final_df = final_df.merge(drug_mechanism[['drug_mechanism|molregno','drug_mechanism|mechanism_of_action','drug_mechanism|tid','drug_mechanism|action_type',]],
                          how='outer',left_on=['activities|molregno','assays|tid'],right_on=['drug_mechanism|molregno','drug_mechanism|tid'])
print(f'added drug mechanism: {final_df.shape}')

## remake molregno column by combining
ind = final_df[(final_df['drug_mechanism|molregno']==final_df['drug_mechanism|molregno'])&          (final_df['activities|molregno']!=final_df['activities|molregno'])].index
final_df['activities_drug_mechanism|molregno']=final_df['activities|molregno'].copy()
final_df.loc[ind,'activities_drug_mechanism|molregno']=final_df.loc[ind,'drug_mechanism|molregno']
print(sum(final_df['activities_drug_mechanism|molregno']==final_df['activities_drug_mechanism|molregno']))
del ind

## remake tid column by combining
ind = final_df[(final_df['drug_mechanism|tid']==final_df['drug_mechanism|tid'])&          (final_df['assays|tid']!=final_df['assays|tid'])].index
final_df['assays_drug_mechanism|tid']=final_df['assays|tid'].copy()
final_df.loc[ind,'assays_drug_mechanism|tid']=final_df.loc[ind,'drug_mechanism|tid']
print(sum(final_df['assays_drug_mechanism|tid']==final_df['assays_drug_mechanism|tid']))


# merge compound informations
final_df = final_df.merge(molecule_dictionary[['molecule_dictionary|molregno', 'molecule_dictionary|pref_name','molecule_dictionary|chembl_id', 'molecule_dictionary|max_phase', 
                                               'molecule_dictionary|molecule_type','molecule_dictionary|oral','molecule_dictionary|parenteral', 'molecule_dictionary|topical',
                                               'molecule_dictionary|black_box_warning','molecule_dictionary|natural_product']],
                          how='left',left_on='activities_drug_mechanism|molregno',right_on='molecule_dictionary|molregno')

final_df = final_df.merge(molecule_atc_classification[['molecule_atc_classification|molregno','molecule_atc_classification|level5']],
                          how='left',left_on='activities_drug_mechanism|molregno',right_on='molecule_atc_classification|molregno')

final_df = final_df.merge(atc_classification[['atc_classification|level1','atc_classification|level2','atc_classification|level3','atc_classification|level4','atc_classification|level5',
                                              'atc_classification|level1_description','atc_classification|level2_description',
                                              'atc_classification|level3_description','atc_classification|level4_description','atc_classification|who_name']],
                          how='left',left_on='molecule_atc_classification|level5',right_on='atc_classification|level5')
print(f'added compound info: {final_df.shape}')


# merge targets
## merging target dataframes first (to link tid with gene symbol)
targets_final = target_dictionary.merge(target_components,how='left',left_on='target_dictionary|tid',right_on='target_components|tid')
targets_final = targets_final.merge(component_synonyms,how='left',left_on='target_components|component_id',right_on='component_synonyms|component_id')
print(targets_final['component_synonyms|syn_type'].value_counts())

## selecting targets which have 'gene symbol'
## syn_type == 'GENE_SYMBOL'
targets_final = targets_final[targets_final['component_synonyms|syn_type']=='GENE_SYMBOL']

## merge
final_df = final_df.merge(targets_final[['target_dictionary|tid','target_dictionary|target_type','target_dictionary|pref_name','target_dictionary|organism','target_dictionary|chembl_id',
                                         'component_synonyms|component_synonym','target_components|component_id']],
                          how='left',left_on='assays_drug_mechanism|tid',right_on='target_dictionary|tid')

print(f'added target info: {final_df.shape}')

final_df.head()


# In[21]:


# add molecular info here::
#final_df = final_df.merge(how='left',left_on='activities_drug_mechanism|molregno',right_on='molecule_atc_classification|molregno')
final_df = final_df.merge(db_x_smiles,how='left',left_on='molecule_dictionary|molregno',right_on='db_x_smiles|molregno')

# Add BIO_ONT to db:
final_df = final_df.merge(db_x_ont, how='left',left_on='assays|bao_format',right_on='db_x_ont|bao_id')


# In[22]:


db_x_COMPONENT_GO.head()


# In[23]:


db_x_GO_CLASSIFICATION.head()


# In[24]:


final_df.columns


# ## Save the data frame above for access to non-human entries

# ## Filtering drugs targeting human molecules

# In[25]:


final_df = final_df[final_df['target_dictionary|organism']=='Homo sapiens']
final_df.shape


# In[26]:


final_df.columns


# ## Add target class

# - Based on [IDG Protein list](https://druggablegenome.net/IDGProteinList)
# - add 'Ion channel' from [HGNC, GID:177](https://www.genenames.org/data/genegroup/#!/group/177)
# - add 'GPCR' from [HGNC, GID:139](https://www.genenames.org/data/genegroup/#!/group/139)
# - add 'NHR' (Nuclear Hormone Receptors) from [HGNC, GID:71](https://www.genenames.org/data/genegroup/#!/group/71)

# In[27]:


# create dictionary for protein classes
idg = pd.read_csv('../resources/IDG_TargetList_Y4.csv')

targetclass_dict={}
for c in set(idg['IDGFamily']):
    targetclass_dict[c]=list(idg[idg['IDGFamily']==c]['Gene'])

ion = pd.read_csv('../resources/HGNC_GID177_Ion-channels.txt',sep='\t')
gpcr = pd.read_csv('../resources/HGNC_GID139_G-protein-coupled-receptors.txt',sep='\t')
nr = pd.read_csv('../resources/HGNC_GID71_Nuclear-hormone-receptors.txt',sep='\t')

targetclass_dict['Ion Channel']=list(set(targetclass_dict['Ion Channel']+list(ion['Approved symbol'])))
targetclass_dict['GPCR']=list(set(targetclass_dict['GPCR']+list(gpcr['Approved symbol'])))
targetclass_dict['NHR']=list(nr['Approved symbol'].unique())
targetclass_dict.keys()


# In[28]:


# assin protein class to each target
def which_class(dictionary, value):
    out='none'
    for k in dictionary.keys():
        if value in dictionary[k]:
            if out=='none':
                out=k
            else:
                out=f'{out};{k}'
    return out

# add target class
final_df['target_class']=final_df['component_synonyms|component_synonym'].copy()
final_df['target_class']=[which_class(targetclass_dict,t) for t in final_df['target_class']]
final_df['target_class'].value_counts()


# In[29]:


final_df.to_csv('./A1_V3_IG_chembl_33_merged_genesymbols_humans.csv')


# In[30]:


final_df.shape


# # read back

# In[6]:


final_df = pd.read_csv('A1_V2_IG_chembl_33_merged_genesymbols_humans.csv',index_col = 0)


# # design a mapper instead of merging the columns for GO terms

# In[39]:


db_x_COMPONENT_GO = db_x_COMPONENT_GO[db_x_COMPONENT_GO['db_x_COMPONENT_GO|component_id'].isin(list(set(final_df['target_components|component_id'])))]
db_x_GO_CLASSIFICATION = db_x_GO_CLASSIFICATION[db_x_GO_CLASSIFICATION['db_x_GO_CLASSIFICATION|go_id'].isin(list(set(db_x_COMPONENT_GO['db_x_COMPONENT_GO|go_id'])))]


# In[48]:


# Remove duplicate rows based on 'component_id' and 'go_id' columns
comps_grouped = db_x_COMPONENT_GO.groupby('db_x_COMPONENT_GO|component_id')['db_x_COMPONENT_GO|go_id'].apply('; '.join).reset_index()
#unique_rows = db_x_COMPONENT_GO.drop_duplicates(subset=['db_x_COMPONENT_GO|component_id', 'db_x_COMPONENT_GO|go_id'])

# Create a dictionary from the unique rows
mapper_comp = dict(zip(comps_grouped['db_x_COMPONENT_GO|component_id'], comps_grouped['db_x_COMPONENT_GO|go_id']))

final_df['db_x_COMPONENT_GO|go_id'] = final_df['target_components|component_id'].map(mapper_comp)
final_df['db_x_COMPONENT_GO|go_id'] = final_df['db_x_COMPONENT_GO|go_id'].astype(str)
# # Create a second mapper:
# # Group by 'db_x_GO_CLASSIFICATION|go_id' and join 'db_x_GO_CLASSIFICATION|path' strings
# paths_grouped = db_x_GO_CLASSIFICATION.groupby('db_x_GO_CLASSIFICATION|go_id')['db_x_GO_CLASSIFICATION|path'].apply('; '.join).reset_index()

# Create a dictionary from the grouped DataFrame
mapper_class = dict(zip(paths_grouped['db_x_GO_CLASSIFICATION|go_id'], paths_grouped['db_x_GO_CLASSIFICATION|path']))

# Map the concatenated paths onto 'final_df' using the 'db_x_COMPONENT_GO|go_id' column
final_df['db_x_GO_CLASSIFICATION|path'] = final_df['db_x_COMPONENT_GO|go_id'].map(mapper_class)

# Function to map and join paths for multiple go_id values
def map_and_join_paths(go_ids, mapper):
    # Split the go_ids by ';', map each to its path using the mapper, and join the resulting paths with '; '
    paths = [mapper.get(go_id.strip()) for go_id in go_ids.split(';') if go_id.strip() in mapper]
    return '; '.join(filter(None, paths))

# Apply the function to each row in 'final_df' to get the joined paths
final_df['db_x_GO_CLASSIFICATION|path'] = final_df['db_x_COMPONENT_GO|go_id'].apply(map_and_join_paths, mapper=mapper_class)


# In[49]:


# merge GO informations
# final_df = final_df.merge(db_x_COMPONENT_GO,how='left',left_on='target_components|component_id',right_on='db_x_COMPONENT_GO|component_id')
# final_df = final_df.merge(db_x_GO_CLASSIFICATION,how='left',left_on='db_x_COMPONENT_GO|go_id',right_on='db_x_GO_CLASSIFICATION|go_id')
print(f'added GO info: {final_df.shape}')


# In[51]:


final_df.head(20)


# ## Save

# In[ ]:


final_df.to_csv('./A1_V3_IG_chembl_33_merged_genesymbols_humans.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', "final_df.to_pickle('./A1_V3_IG_chembl_33_merged_genesymbols_humans.pkl')")


# In[ ]:


len(final_df)


# ## Session info

# In[ ]:


import session_info
session_info.show()


# In[ ]:


final_df.head(20)


# In[ ]:


final_df.loc[final_df['molecule_dictionary|chembl_id'].isin(['CHEMBL1004'])]


# In[ ]:


final_df.loc[final_df['molecule_dictionary|chembl_id'].isin(['CHEMBL1004']),['drug_mechanism|molregno', 'drug_mechanism|mechanism_of_action',
       'drug_mechanism|tid', 'drug_mechanism|action_type',
       'activities_drug_mechanism|molregno', 'assays_drug_mechanism|tid',]]


# In[65]:


final_df


# # Filter database

# In[ ]:


import pandas as pd
import drug2cell as d2c


# Load the human targets ChEMBL data frame created in the initial parsing notebook.

# In[ ]:


original = final_df.copy() #pd.read_pickle("chembl_30_merged_genesymbols_humans.pkl")


# Drug2cell's filtering functions allow for subsetting the pchembl threshold for each category of a column of choice. We'll be using the `target_class` column, and basing our values on https://druggablegenome.net/ProteinFam

# In[ ]:


#pChEMBL is -log10() as per https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions#what-is-pchembl
thresholds_dict={
    'none':7.53, #30nM
    'NHR':7, #100nM
    'GPCR':7, #100nM
    'Ion Channel':5, #10uM
    'Kinase':6, #1uM
}


# We'll add some more criteria to the filtering. For a comprehensive list of available options, consult the documentation.

# In[ ]:


filtered_df = d2c.chembl.filter_activities(
    dataframe=original,
    drug_max_phase=4,
    assay_type='F',
    add_drug_mechanism=True,
    remove_inactive=True,
    include_active=True,
    pchembl_target_column="target_class",
    pchembl_threshold=thresholds_dict
)
print(filtered_df.shape)


# In[ ]:


filtered_df.to_csv('./A1_V3_IG_chembl_33_filtered_merged_genesymbols_humans.csv')


# In[64]:


filtered_df


# Now that we have our data frame subset to the drugs and targets of interest, we can convert them into a dictionary that can  be used by drug2cell. The exact form distributed with the package was created like so:

# In[ ]:


chembldict = d2c.chembl.create_drug_dictionary(
    filtered_df,
    drug_grouping='ATC_level'
)
chembldict['B03']


# In[ ]:


# with open('my_dict.pkl', 'wb') as pickle_file:
#     pickle.dump(my_dict, pickle_file)


# This results in a nested dictionary structure - a dictionary of categories, holding dictionaries of drugs, holding lists of targets. Drug2cell knows how to operate with this sort of structure as well as its normal groups:targets dictionary, but you need to specify `nested=True` in the scoring/enrichment/overrepresentation functions whenever you pass this structure.

# In[ ]:




