
import pandas as pd
import time
import streamlit as st
import plotly.express as px

import pandas as pd
import numpy as np
from pymorphy2 import MorphAnalyzer
from collections import Counter
import seaborn as sns
import pickle
from datetime import date
import requests
import re

morph = MorphAnalyzer()
st.title('Расчет вреда накопления в организме человека «пищевой химии»')

cat_diseases = {'аллерген',
 'аллергия',
 'анафилактический',
 'антиоксидант',
 'астма',
 'бессоница',
 'болезнь',
 'боли',
 'вес',
 'глаза',
 'головная',
 'давление',
 'дахытельные',
 'депрессия',
 'дерматит',
 'диарея',
 'дыхательные',
 'желудок',
 'живот',
 'злокачественные',
 'зрение',
 'зубы',
 'зуд',
 'избыток',
 'имунная',
 'инсульт',
 'интоксикация',
 'канцероген',
 'кариес',
 'кишечник',
 'кожа',
 'кости',
 'крапивница',
 'легкие',
 'летальные',
 'метаболизм',
 'метеоризм',
 'мигрень',
 'мутация',
 'насморк',
 'недомогание',
 'нервная',
 'обезвоживание',
 'ожоги',
 'онкология',
 'опухоли',
 'осмотическое',
 'остеопороз',
 'отек',
 'отеки',
 'отравление',
 'печень',
 'пищеварение',
 'повреждение',
 'повышение',
 'почки',
 'приступы',
 'психика',
 'пузырьки',
 'раздражение',
 'рак',
 'раны',
 'расстройство',
 'рахит',
 'рвота',
 'сердевно-сосудистая',
 'сердце',
 'слабительное',
 'слабость',
 'слизистая',
 'слизистые',
 'смертельно',
 'снижение',
 'сыпь',
 'сырь',
 'тахикардия',
 'тонус',
 'тошнота',
 'удушье',
 'усталость',
 'ухудшение',
 'фиброз',
 'хим',
 'холестерин',
 'щитовидка',
 'экзема',
 'язва',
 'язвы'}

def make_grafik(info):
    start_name = [x[0] for x in info]
    cur = [change_dic[x[0]] for x in info]
    amount = [x[1] for x in info]
    start_dates = [(date.today() - x[2]).days for x in info]
    end_dates = [0 if x[4] else (date.today() - x[3]).days for x in info]

    # string = change_dic[string]
    
    # string = preprocess('Молочный шоколад (сахар, какао тертое, сухое обезжиренное молоко, какао масло, эквивалент масла какао (масло пальмовое, масло ши), лактоза, молочный жир, эмульгатор (соевый лецитин), ароматизаторы (ванилин, масляная кислота)); сахар, крахмал кукурузный, глюкозный сироп (кукурузный, пшеничный), загуститель (декстрин), крахмал рисовый, красители (Е120, куркумин, каротины, Е133, карбонат кальция), глазирователь (воск карнаубский), жир специального назначения (масло кокосовое, масло пальмоядровое) е559.')
    # string = preprocess('Вода, краситель сахарный колер IV, регулятор кислотности кислота ортофосфорная, подсластители (цикламат натрия, ацесульфам калия, сукралоза, сахаринат натрия), ароматизаторы, консервант бензоат натрия, кофеин (не более 150 мг/л).')
    
    counts = []
    bad_make = []
    for string in cur:
    
        
        string = preprocess(string)
        components_idx = []

        for idx, component in enumerate(cur_components):
            # также меняем английскую E на русскую
            for c in component.split(','):
                if c in string or c.replace('е', 'е') in string: 
                    components_idx.append(idx)
                    break

        info = []
        info_bad = []
        for id in components_idx:
            a = df.Опасность_new.tolist()[id]
            if a > 0:
                info_bad.append(a)
            
            a = df.Влияния.tolist()[id]
            if type(a) != str:
                continue
            for x in a.split():
                info.append(x.replace(',', ''))
                break

        counts.append(len(info))
        bad_make.append(sum(info_bad))

    a = pd.DataFrame({
        'Товар': start_name,
        'Кол-во вредных элементов': counts,
        'Относительный вред': bad_make,
        'Начало': start_dates,
        'Конец': end_dates
    })

    cur_sum_info = [0 for _ in range(31)]
    cur_count_info = [0 for _ in range(31)]

    for x in range(len(a)):
        for k in range(end_dates[x], start_dates[x]+1):
            if k <= 30:
                cur_sum_info[k] += bad_make[x] * amount[x] / 7
                cur_count_info[k] += counts[x] * amount[x] / 7
                

    change_df = pd.DataFrame({
        'Сколько дней назад': [x for x in range(31)],
        'Кол-во вреда': cur_sum_info,
        'Кол-во потребляемых плохих химикатов': cur_count_info
    })
    st.subheader('Информация о кол-ве "пищевой химии" в продукте')    
    st.dataframe(a.drop(['Начало', 'Конец'], axis=1), use_container_width=False)

    st.subheader('Изменение потребления "пищевой химии"')

    if selected_to_show == 'Кол-во вреда':
        st.line_chart(data=change_df, x='Сколько дней назад', y='Кол-во вреда')
    elif selected_to_show == 'Кол-во вредных химикатов':
        st.line_chart(data=change_df, x='Сколько дней назад', y='Кол-во потребляемых плохих химикатов')

def get_color(t):
    t = t.lower()
    if 'низк' in t or 'нулевая' in t or 'безопас' in t or 'безвредн' in t:
        return 'green'
    elif 'средняя' in t:
        return 'orange'
    else:
        return 'red'


df = pd.read_excel('data/tmp copy.xlsx', index_col=0)
df['Опасность_new'] = df['Опасность'].apply(lambda x: ' '.join(x.split(',')[0].split()) if type(x)==str else x)

change_dang = {
    'Нулевая': 0,
    'Безвредный': 0,
    'Безопасный': 0,
    'Безопасен': 0,
    'Очень низкая': 1,
    'Низкая': 2,
    'Низкий': 2,
    'Не безопасен': 4,
    'Средняя': 5,
    'Опасен': 7,
    'Опасная': 7,
    'Опасный': 7,
    'Высокая': 8,
    'Высока': 8,
    'Очень опасен': 9,
    'Очень опасный': 9,
    'Высокая опасность': 9,
    'Очень высокая': 9,
    'Запрещен': 10,
}

df['Опасность_new'] = df['Опасность_new'].replace(change_dang)


def preprocess(text):
    
    text = str(text)
    text = text.lower().strip()
    text=re.compile('<.*?>').sub('', text)
    text = re.sub('\s+', ' ', text)
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\s+',' ',text)
    return text

def show_my():
    st.subheader('Мои товары')
    st.write(tmp_df)

all_components = []
cur_components = []
for full_name in df['Е, название'].values:
    full_name = str(full_name)
    all_components = all_components + full_name.split(',')
    cur_components.append(','.join([preprocess(x) for x in full_name.split(',')]))
all_components = [preprocess(x) for x in all_components]
df['Е, название_new'] = cur_components

all_info = pickle.load(open('data/all_info.pkl', 'rb'))
example = pd.read_csv('data/new_full_v2.csv')
change_dic = {example['name'].tolist()[x]:example['components'].tolist()[x] for x in range(len(example))}
st.subheader('Выберите действие')            
select_doing = st.selectbox('Что вы хотите делать', ['Добавить товар', 'Остановить потребление',
                                                     'Посмотреть статистику в целом', 'Статистика для разных групп',
                                                     'Найти самые вредные продукты', 'Посмотреть информацию о конкретном продукте'])

tmp_df = pd.DataFrame({
    'Название': [x[0] for x in all_info],
    'Кол-во': [x[1] for x in all_info],
    'Дата начала': [x[2] for x in all_info],
    'Дата окончания': [x[3] for x in all_info],
    'Будет продолжаться': [x[4] for x in all_info]
})

if select_doing == 'Добавить товар':
    show_my()
    selected_category = st.selectbox('Выбрать категорию продукта', example['category'].unique())
    selected_sex = st.selectbox("Выбрать название продукта ", example[example['category']==selected_category]['name'].unique())
    selected_count = st.number_input('Сколько раз в неделю вы его употребляете')
    select_start = st.date_input('Дата начала')
    select_end = st.date_input('Дата окончания')
    select_future = st.checkbox('Буду продолжать в будущем')
    but = st.button('Добавить товар')

    if but and selected_count > 0:
        all_info.append([selected_sex, selected_count, select_start, select_end, select_future])
        pickle.dump(all_info, open('data/all_info.pkl', 'wb'))
    st.button('Обновить страничку')

elif select_doing == 'Посмотреть статистику в целом':
    res = st.button('Удалить все товары')
    if res:
        pickle.dump([], open('data/all_info.pkl', 'wb'))
    st.button('Обновить страничку')
    selected_to_show = st.selectbox('Что визуализировать?', ['Кол-во вреда', 'Кол-во вредных химикатов в продукте'])
    make_grafik(all_info)
    
elif select_doing == 'Остановить потребление':
    show_my()
    if len(tmp_df) > 0:
        select_to_delete = st.selectbox('Выбрать товар, который вы перестали употреблять', tmp_df['Название'].unique())
        select_end_date = st.date_input('Дата окончания')
        remove = st.button('Удалить данный товар')
        if remove:
            for x in range(len(tmp_df)-1, -1, -1):
                if select_to_delete == all_info[x][0]:
                    all_info[x][3] = select_end_date
                    all_info[x][4] = False
                    pickle.dump(all_info, open('data/all_info.pkl', 'wb'))
                    break
    else:
        st.write('Нет продуктов для удаления')
    st.button('Обновить страничку')
    
elif select_doing == 'Найти самые вредные продукты':
    change_dic = {example['name'].tolist()[x]:example['components'].tolist()[x] for x in range(len(example))}
    cat_dic = {example['name'].tolist()[x]:example['category'].tolist()[x] for x in range(len(example))}
    cur = [change_dic[x[0]] for x in all_info if cat_dic]

    counts = []
    info_unique = []
    info_sum = []
    for string in cur:
        string = preprocess(string)
        components_idx = []
        for idx, component in enumerate(cur_components):
            for c in component.split(','):
                if c in string or c.replace('е', 'е') in string: 
                    components_idx.append(idx)
                    break

        info = []
        cur_info_sum = []
        for id in components_idx:
            a = df.Влияния.tolist()[id]
            num = df.Опасность_new.tolist()[id]
            if type(a) != str:
                continue
            for idx, x in enumerate(a.split(',')):
                if idx == 0:
                    info.append(x.replace(',', ''))
                    if num > 0:
                        cur_info_sum.append(num)
                info_unique.append(x.replace(',',''))
    
        counts.append(len(info))
        info_sum.append(sum(cur_info_sum))

    info_unique = set(info_unique)
    a = pd.DataFrame({
        'Товар': [x[0] for x in all_info],
        'Кол-во вредных элементов': counts,
        'Относительный вред': info_sum
    })
    a = a[a['Кол-во вредных элементов'] > 0]
    st.write(a)
    select_cat = st.selectbox('По какой категории?', ['Все']+list(info_unique))
    select_choice = st.selectbox('Что выводить?', ['Кол-во веществ', 'Относительный вред'])
    
    if select_cat == 'Все':

        if len(a) > 0:
            if select_choice == 'Кол-во веществ':
                st.bar_chart(data=a, y='Кол-во вредных элементов', x='Товар')
            else:
                st.bar_chart(data=a, y='Относительный вред', x='Товар')
        else:
            st.write('Вредные вещества не найдены')
        st.button('Обновить страничку')
        
    else:
        cur = [change_dic[x[0]] for x in all_info]
        
        counts = []
        info_sum = []
        for string in cur:
            string = preprocess(string)
            components_idx = []

            for idx, component in enumerate(cur_components):
                for c in component.split(','):
                    if c in string or c.replace('е', 'е') in string: 
                        components_idx.append(idx)
                        break

            info = []
            cur_info_sum = []
            for id in components_idx:
                a = df.Влияния.tolist()[id]
                num = df.Опасность_new.tolist()[id]
                if type(a) != str or select_cat not in [x.replace(',', '') for x in a.split(',')]:
                    continue
                for idx, x in enumerate(a.split(',')):
                    if idx == 0:
                        info.append(x.replace(',', ''))
                        if num > 0:
                            cur_info_sum.append(num)
                    
            counts.append(len(info))
            info_sum.append(sum(cur_info_sum))

        a = pd.DataFrame({
            'Товар': [x[0] for x in all_info],
            'Кол-во вредных элементов': counts,
            'Относительный вред': info_sum
        })

        a = a[a['Кол-во вредных элементов'] > 0]
        st.dataframe(a)

        if len(a) > 0:
            st.bar_chart(data=a, y='Кол-во вредных элементов', x='Товар') #, width=0, height=0, use_container_width=True)
        else:
            st.write('Вредные вещества не найдены')
        st.button('Обновить страничку')


elif select_doing == 'Статистика для разных групп':
    change_dic = {example['name'].tolist()[x]:example['components'].tolist()[x] for x in range(len(example))}
    info = []
    for string in [x[0] for x in all_info]:
        string = preprocess(change_dic[string])

        components_idx = []
        for idx, component in enumerate(cur_components):
            for c in component.split(','):
                if c in string or c.replace('е', 'е') in string: 
                    components_idx.append(idx)
                    break


        for id in components_idx:
            a = df.Влияния.tolist()[id]
            if type(a) != str:
                continue
            for x in a.split(','):
                info.append(x)

    count = Counter(info)

    a = pd.DataFrame({
        'Вред': count.keys(),
        'Кол-во элементов': count.values()
    })
    
    if len(a) > 0:
        st.bar_chart(data=a, y='Кол-во элементов', x='Вред')
    else:
        st.write('Вредные вещества не найдены')
    st.button('Обновить страничку')
    
    
elif select_doing == 'Посмотреть информацию о конкретном продукте':
    select_inventory = st.selectbox('Откуда товар', ['Все предметы', 'В наших продуктах'])
    
    if select_inventory == 'Все предметы':
        selected_category = st.selectbox('Выбрать категорию товара', example['category'].unique())
        selected_tovar = st.selectbox("Выбрать название товара", example[example['category']==selected_category]['name'].unique())
        
        change_dic = {example['name'].tolist()[x]:example['components'].tolist()[x] for x in range(len(example))}
        path_dic = {example['name'].tolist()[x]:example['picture'].tolist()[x] for x in range(len(example))}
        info = []
        string = preprocess(change_dic[selected_tovar])
        
        components_idx = []

        for idx, component in enumerate(cur_components):
            for c in component.split(','):
                if c in string or c.replace('е', 'е') in string: 
                    components_idx.append(idx)
                    break
                
        img = requests.get(path_dic[selected_tovar])
        st.image(img.content, width=400)
        weights = np.argsort([df.Опасность_new[x] for x in components_idx])
                
        for tmp in range(len(components_idx)):
            id = components_idx[list(reversed(weights))[tmp]]
            c = st.container()
            name = df['Е, название'].tolist()[id]
            c.subheader(name)
            dang = df['Опасность'].tolist()[id]
            color = get_color(dang)
            c.markdown('**Опасность:** ' + f':{color}[{dang}]')
            vl = df['Влияние на организм'].tolist()[id]
            c.markdown('**Влияние на организм:** '+vl)
            st.divider()
    elif select_inventory == 'В наших продуктах':
        selected_tovar = st.selectbox("Выбрать название товара", set([x[0] for x in all_info]))
        change_dic = {example['name'].tolist()[x]:example['components'].tolist()[x] for x in range(len(example))}
        path_dic = {example['name'].tolist()[x]:example['picture'].tolist()[x] for x in range(len(example))}
        info = []
        string = preprocess(change_dic[selected_tovar])
        components_idx = []
        for idx, component in enumerate(cur_components):
            # также меняем английскую E на русскую
            for c in component.split(','):
                if c in string or c.replace('е', 'е') in string: 
                    components_idx.append(idx)
                    break
                
        img = requests.get(path_dic[selected_tovar])
        st.image(img.content)
        weights = np.argsort([df.Опасность_new[x] for x in components_idx])
        for tmp in range(len(components_idx)):
            id = components_idx[list(reversed(weights))[tmp]]
            c = st.container()
            name = df['Е, название'].tolist()[id]
            c.subheader(name)
            dang = df['Опасность'].tolist()[id]
            color = get_color(dang)
            c.markdown('**Опасность:** ' + f':{color}[{dang}]')
            vl = df['Влияние на организм'].tolist()[id]
            c.markdown('**Влияние на организм:** '+vl)
            st.divider()
