import warnings
import pyreadstat
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
from scipy.stats import chi2_contingency
from dython.nominal import associations
import ppscore as pps
warnings.simplefilter("ignore")
pn.extension('plotly', sizing_mode="stretch_width", raw_css=[open("styling.css").read()])

def read_df():
    df, meta = pyreadstat.read_sav('data.sav', encoding="ISO-8859-1", apply_value_formats=True)
    df.drop([c for c in df.columns if any(c.startswith(x) for x in ['perception', 'gad', 'important', 'PHQ', 'IES'])], axis=1, inplace=True)
    df.replace('', np.nan, inplace=True)
    df.rename(columns={'classdepression21': 'depression_2021',
                       'classanxiety21': 'anxiety_2021',
                       'classptsd21': 'PTSD_2021',
                       'classedepression23': 'depression_2023',
                       "classeanxiety23": 'anxiety_2023',
                       "classeptsd23": "PTSD_2023",
                       'Interpretationdepression': 'PHQ9_2021',  # 
                       'Interpretationanxiete': 'GAD7_2021',  # 
                       'Interpretationestpt': 'IES-R_2021',  # 
                       'interpretationphq': 'PHQ9_2023',  # 
                       "interpretationgad": 'GAD7_2023',  # 
                       "inter1ies": 'IES-R_2023',  # 
                       'statutmartial': 'Martial_Status',  # 
                       'niveaudeducation': "Educational_level",  # 
                       'nombredepersonne': 'nb_person',  # 
                       'zonedhabitat': "habitat_zone",  # 
                       'niveausocioeconomique': 'social_level',  # 
                       'atcdssomatiques': 'illness_history',  # 
                       'atcdspsychiatriquespersonnels': 'personal_psy_illness_history',
                       'datedelifectioncovid': "covid_infection_date",
                       'assistancerespiratoire': 'respiratory_assistance',  # 
                       'lieudisolement': "quanrantine_area",  # 
                       'avezvscontinuerdetravailleraveclinfectioncovid': "work_while_infected",# 
                       'dureedarretdetravail': "sick_leave_days",  # 
                       'utilisezvslesmesuresdhygieneetdeprotectiondanslaviequotidienne': "hygiene_measures",  # 
                       'couvrezsystÃ©matiquementlabouche': 'mouth_covered',  # 
                       'appliquezvslesreglesdedistanciationphysique': 'distancing_measrues', # 
                       'avezvsreduitvoscontacts': 'reduced_contact',  # 
                       'lasourcedinformationdurantlapandemie': "Covid_information_source", # 
                       'avezvsconstatÃ©uneaugmentationdelachargedetravail': 'increased_workload', #
                       'avezvsconsulterauparavantunpsy' : 'psy_consulted', 
                       'avezvseubesoindunsoutienpsychologique': "psy_help_need",  # 
                       'avezvseubesoinduneconsultationpsychiatrique': "psy_consult_need",
                       'prenezvsdesmedicamentspsychotropes' : 'psy_drugs',
                       'avezvsconsulteupsydepuisledebutdelepidemie': "psy_consulted_since_covid", # 
                       'atcdsomatique': 'illnesses_history',
                       'atcdspsyfamiliaux' : 'family_psy_illness_history',
                       'avezvssentiquevsavezstigmatise' : 'stigmatized', 
                       'pcrdecotrolepourreprise' : 'pcr_work_return',
                       'avezvousprisdesdispositionsdelogementsspecialesdurantlapandÃ©mie' : 'housing_special_measures',
                       'numtel': 'number',  # 
                       'V1': 'name',  # ,
                       'Genre': 'gender',
                       'longcovidsym': 'long_covid_symptoms'}, inplace=True)
    df['number'] = df.number.apply(lambda x: np.nan if len(str(x)) < 8 or not x.isdigit() else x[:8].strip())
    df["covid_infection_date"] = pd.to_datetime(df.covid_infection_date).dt.to_period('M')
    df['age'] = df.age.astype(np.int8)
    df['sick_leave_days'] = df.sick_leave_days.astype(np.float32)
    df['nb_person'] = df.nb_person.astype(np.int8)
    df['age_class'] = pd.cut(df.age.astype('int8'), bins=[18, 25, 35, 45, 55, 65, 75, 85])
    df.drop(['name', 'number'], axis=1, inplace=True)
    return df

def crosstab():
    columns = ['depression_2023', 'anxiety_2023', 'PTSD_2023', 'gender', 'age_class','social_level', 'Educational_level', 'Martial_Status', 
           'long_covid_symptoms']
    # define the select widgets
    select1 = pn.widgets.Select(name='X', options=columns)
    select2 = pn.widgets.Select(name='Y', options=columns)
    select3 = pn.widgets.Select(name='Z', options=columns)
    # function to update the crosstab based on the selected columns
    @pn.depends(select1.param.value, select2.param.value, select3.param.value)
    def update_crosstab(selected_column1, selected_column2, selected_column3):
        cross_tab = pd.crosstab([read_df()[selected_column1], read_df()[selected_column2]], read_df()[selected_column3],
                                margins=True, margins_name="Total", normalize='all').round(4)*100
        return cross_tab.style.format("{:.2f}%").background_gradient(cmap='cubehelix') 
    # create a panel with the select widgets and the crosstab
    return pn.Column(pn.Row(select1, select2, select3, css_classes=['half-width']), update_crosstab, sizing_mode='stretch_width', 
                     css_classes=['flex-1', 'bordered'])
    
def scatter():
    columns_xy = ['age', 'PHQ9_2023', 'GAD7_2023', 'IES-R_2023']
    columns_hue = [None, 'gender', 'habitat', 'psy_help_need']
    # define the select widgets
    select1 = pn.widgets.Select(name='X', options=columns_xy)
    select2 = pn.widgets.Select(name='Y', options=columns_xy)
    select3 = pn.widgets.Select(name='hue', options=columns_hue)
    # Function to update the crosstab based on the selected columns
    @pn.depends(select1.param.value, select2.param.value, select3.param.value)
    def update_scatterplot(selected_column1, selected_column2, selected_column3):
        fig = px.scatter(read_df(), x=selected_column1, y=selected_column2, color=selected_column3, trendline="ols")#, color_discrete_sequence=px.colors.qualitative.Antique)
        fig.update_layout(template='presentation',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(t=0, l=10, r=0, b=10),
                          xaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                          yaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                          legend=dict(title_font=dict(family='Rockwell', color='white', size=12), font=dict(family='Rockwell', color='white', size=10)))
        return pn.pane.Plotly(fig, sizing_mode='stretch_both')
    # Create a Panel with the Select widgets and the crosstab
    return pn.Column(pn.Row(select1, select2, select3, css_classes=['half-width']), update_scatterplot, css_classes=['flex-13','bordered'])

def barplot():
    columns_hue = [None, 'gender', 'habitat', 'psy_help_need']
    columns_x = ['depression_2023', 'anxiety_2023', 'PTSD_2023', 'gender', 'age_class','social_level', 'Educational_level', 'Martial_Status']
    columns_y = ['PHQ9_2023', 'GAD7_2023', 'IES-R_2023']
    # Define the select widgets
    select_x = pn.widgets.Select(name='X', options=columns_x)
    select_y = pn.widgets.Select(name='Y', options=columns_y)
    select_hue_1 = pn.widgets.Select(name='hue 1', options=columns_hue) 
    select_hue_2 = pn.widgets.Select(name='hue 2', options=columns_hue)
    # Function to update the crosstab based on the selected columns

    @pn.depends(select_x.param.value, select_y.param.value, select_hue_1.param.value, select_hue_2.param.value)
    def update_barplot(selected_column1, selected_column2, selected_column3, selected_column4):
        fig = px.histogram(read_df(), x=selected_column1, y=selected_column2, color=selected_column3, pattern_shape=selected_column4, barmode='group', histfunc='avg',
                           text_auto='.2f')
        fig.update_layout(template='presentation',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(t=5, l=10, r=0, b=10),
                          xaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                          yaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                          legend=dict(title_font=dict(family='Rockwell', color='white', size=12), font=dict(family='Rockwell', color='white', size=10)))
        fig.update_traces(textangle=0, textposition="outside", cliponaxis=False, textfont=dict(family='Rockwell', color='white', size=10))
        return pn.pane.Plotly(fig, sizing_mode='stretch_both')
    # Create a Panel with the Select widgets and the crosstab
    return pn.Column(pn.Row(select_x, select_y, select_hue_1, select_hue_2, css_classes=['half-width']), update_barplot)

def countplot():
    columns_hue = [None, 'gender', 'habitat', 'psy_help_need']
    columns_x = ['depression_2023', 'anxiety_2023', 'PTSD_2023']
    # Define the select widgets
    select_x = pn.widgets.Select(name='X', options=columns_x)
    select_hue_1 = pn.widgets.Select(name='hue 1', options=columns_hue)
    select_hue_2 = pn.widgets.Select(name='hue 2', options=columns_hue)
    # Function to update the crosstab based on the selected columns

    @pn.depends(select_x.param.value, select_hue_1.param.value, select_hue_2.param.value)
    def update_countplot(selected_column1, selected_column2, selected_column3):
        fig = px.histogram(read_df(), x=selected_column1, color=selected_column2, pattern_shape=selected_column3, barmode='group', histfunc='count', histnorm='percent',
                           text_auto='.2f')
        fig.update_layout(template='presentation',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(t=5, l=10, r=0, b=10),
                          xaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                          yaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                          legend=dict(title_font=dict(family='Rockwell', color='white', size=12), font=dict(family='Rockwell', color='white', size=10)))
        fig.update_traces(textangle=0, textposition="outside", cliponaxis=False, textfont=dict(family='Rockwell', color='white', size=10))
        return pn.pane.Plotly(fig, sizing_mode='stretch_both')

    # Create a Panel with the Select widgets and the crosstab
    return pn.Column(pn.Row(select_x, select_hue_1, select_hue_2, css_classes=['half-width']), update_countplot)

def histplot():
    columns_hue = [None, 'gender', 'habitat', 'psy_help_need']
    columns_x = ['PHQ9_2023', 'GAD7_2023', 'IES-R_2023']
    # Define the select widgets
    select_x = pn.widgets.Select(name='X', options=columns_x)
    select_hue_1 = pn.widgets.Select(name='hue 1', options=columns_hue)
    select_hue_2 = pn.widgets.Select(name='hue 2', options=columns_hue)
    # Function to update the crosstab based on the selected columns
    @pn.depends(select_x.param.value, select_hue_1.param.value, select_hue_2.param.value)
    def update_histplot(selected_column1, selected_column2, selected_column3):
        fig = px.histogram(read_df(), x=selected_column1, color=selected_column2, pattern_shape=selected_column3, histnorm='probability', barmode='group',
                           nbins=int(np.ptp(read_df()[selected_column1].dropna())), text_auto='.2f')
        fig.update_layout(template='presentation',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', bargap=.1,
                          margin=dict(t=5, l=10, r=0, b=10),
                          xaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                          yaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                          legend=dict(title_font=dict(family='Rockwell', color='white', size=12), font=dict(family='Rockwell', color='white', size=10)))
        fig.update_traces(textangle=0, textposition="outside", cliponaxis=False, textfont=dict(family='Rockwell', color='white', size=10))
        return pn.pane.Plotly(fig, sizing_mode='stretch_both')
    # Create a Panel with the Select widgets and the crosstab
    return pn.Column(pn.Row(select_x, select_hue_1, select_hue_2, css_classes=['half-width']), update_histplot)

dark_gray_hex_colors = [
    '#0A0A0A',  # Very dark gray
    '#141414',
    '#1E1E1E',
    '#282828',
    '#323232',
    '#3C3C3C'   # Dark gray
]

def treemap():
    columns_xy = ['depression_2023', 'anxiety_2023', 'PTSD_2023', 'long_covid_symptoms']
    # define the select widgets
    select1 = pn.widgets.Select(name='First Level', options=columns_xy)
    select2 = pn.widgets.Select(name='Second Level', options=columns_xy)
    # Function to update the crosstab based on the selected columns
    @pn.depends(select1.param.value, select2.param.value)
    def update_mosaic(selected_column1, selected_column2):
        if selected_column1 == selected_column2:
            fig = px.treemap(read_df().dropna(subset=[selected_column1, selected_column2]), path=[px.Constant("All Levels"), selected_column1], color=selected_column1,
                             hover_data=[selected_column1], color_discrete_sequence=dark_gray_hex_colors)
        else:
            fig = px.treemap(read_df().dropna(subset=[selected_column1, selected_column2]), path=[px.Constant("All Levels"), selected_column1, selected_column2], 
                             color=selected_column2, hover_data=[selected_column1, selected_column2], color_discrete_sequence=dark_gray_hex_colors)
        fig.update_layout(template='presentation', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, l=25, r=0, b=0))
        return pn.pane.Plotly(fig, sizing_mode='stretch_both')
    #create a panel with the Select widgets and the crosstab      
    return pn.Column(pn.Row(select1, select2), update_mosaic, css_classes=['flex-1', 'bordered'])

df_corr = read_df().drop(['depression_2021', 'anxiety_2021', 'PTSD_2021', 'depression_2023', 'anxiety_2023', "PTSD_2023", 'personal_psy_illness_history',
                   'mouth_covered', 'distancing_measrues', 'age_class', 'illness_history', 'siouiprÃ©cisezledgretenu', 'sioui', 'Symptomes', 
                   'siouipreciserlestttprescrits', "hygiene_measures", 'sÃ©quellesdelamaladie', 'illnesses_history', 'diagnosticretenupar',
                   'traitementpris'], axis=1).astype('category')   

df_corr_id = hash(pd.util.hash_pandas_object(df_corr).sum())

def calculate_pvalue_matrix(df):
    columns=df.columns
    pvalue_matrix = pd.DataFrame(index=columns, columns=columns)
    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                pvalue_matrix.loc[col1, col2] = 1
            else:
                _, p, _, _ = chi2_contingency(pd.crosstab(df[col1], df[col2]))
                pvalue_matrix.loc[col1, col2] = p
    return pvalue_matrix.astype(float)

@pn.cache
def corr_matrices(df, df_id):
    return {'chi' : associations(df, compute_only=True, multiprocessing=True)['corr'],
            'pps' : pps.matrix(df, invalid_score=0)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore'),
            'pvalue' : calculate_pvalue_matrix(df).round(4)}
    
def corr_pvalue():
    fig = px.imshow(corr_matrices(df_corr, df_corr_id)['pvalue'], text_auto=".2f", color_continuous_scale='cividis', aspect="auto") 
    fig.update_layout(template='presentation',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=5, l=90, r=0, b=90),
                      xaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                      yaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                      legend=dict(title_font=dict(family='Rockwell', color='white', size=12), font=dict(family='Rockwell', color='white', size=10)))
    return pn.pane.Plotly(fig, sizing_mode='stretch_both')

def corr_chi():
    fig = px.imshow(corr_matrices(df_corr, df_corr_id)['chi'], text_auto=".2f", color_continuous_scale='cividis', aspect="auto")
    fig.update_layout(template='presentation',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(t=5, l=90, r=0, b=90),
                          xaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                          yaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                          legend=dict(title_font=dict(family='Rockwell', color='white', size=12), font=dict(family='Rockwell', color='white', size=10)))
    return pn.pane.Plotly(fig, sizing_mode='stretch_both')

def corr_pps():
    fig = px.imshow(corr_matrices(df_corr, df_corr_id)['pps'], text_auto=".2f", color_continuous_scale='cividis', aspect="auto") 
    fig.update_layout(template='presentation',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=5, l=90, r=0, b=90),
                      xaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                      yaxis=dict(title_text='', tickfont=dict(family='Rockwell', color='white', size=9)),
                      legend=dict(title_font=dict(family='Rockwell', color='white', size=12), font=dict(family='Rockwell', color='white', size=10)))
    return pn.pane.Plotly(fig, sizing_mode='stretch_both')

def main():
    template = pn.template.SlidesTemplate(site="Faculté de Médecine de Tunis", title="Covid 19 Mental health Statiscal Dashboard", theme=pn.template.DarkTheme)
    template.main.extend([pn.Row(pn.Column(scatter(), crosstab(), css_classes=['flex-3']), treemap())])
    template.main.append(pn.Row(pn.Column(pn.Tabs(('countplot', countplot), ('barplot', barplot()), ('histplot', histplot()), 
                                                  css_classes=['bordered']),
                                          crosstab()),  
                                pn.Tabs(('chi-2', corr_chi()), 
                                        ('pps', corr_pps()),
                                        ('p-value', corr_pvalue()),
                                        css_classes=['bordered'])))
    pn.serve(template, show=True)

if __name__ == '__main__':
    # multiprocessing.freeze_support()  # Uncomment this line if you're freezing your script with py2exe or similar
    main()