"""Build the research article from frozen experiment evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle




NAMES = {
    'popularity': 'Популярность', 'itemknn': 'ItemKNN', 'baseline': 'GPT-2',
    'matched_control': 'KL + CE, без PG', 'raw_h5': 'RL без ограничений',
    'anchored_h1': 'RL + KL + CE, H=1', 'anchored_h5': 'RL + KL + CE, H=5',
    'appended_h5': 'RM с добавлением истории', 'conservative_h5': 'Консервативная награда',
    'retention_h5': 'Доля + возвращаемость', 'duration_neutral_h5': 'Вычитание тренда длительности',
    'seconds_h5': 'Ожидаемые секунды', 'reranker': 'Переранжирование RM',
}


def number(value, digits=4):
    return '-' if value is None or not np.isfinite(value) else f'{value:.{digits}f}'


def pct(value, digits=1, signed=False):
    if value is None or not np.isfinite(value):
        return '-'
    return f'{100 * value:+.{digits}f}%' if signed else f'{100 * value:.{digits}f}%'


def pvalue(value):
    return '&lt;0,001' if value < .001 else number(value, 3)


def groups(result):
    grouped = {}
    for name, model in result['models'].items():
        grouped.setdefault(model['family'], []).append(model)
    return grouped


def mean(result, family, metric, split='test'):
    values = [entry[split].get(metric) for entry in groups(result)[family] if split in entry]
    values = [value for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def mean_sd(result, family, metric, split='test', digits=4):
    values = [entry[split].get(metric) for entry in groups(result)[family] if split in entry]
    values = [value for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
    if not values:
        return '-'
    average = number(float(np.mean(values)), digits)
    return average + (' ± ' + number(float(np.std(values, ddof=1)), digits) if len(values) > 1 else '')


def relative(result, family, metric, split='test', comparator='baseline'):
    base = mean(result, comparator, metric, split)
    value = mean(result, family, metric, split)
    return value / base - 1 if base and value is not None else None


def register_fonts():
    directories = [Path('/System/Library/Fonts/Supplemental'), Path('/usr/share/fonts/truetype/msttcorefonts')]
    files = ['Times New Roman.ttf', 'Times New Roman Bold.ttf', 'Times New Roman Italic.ttf', 'Times New Roman Bold Italic.ttf']
    directory = next((folder for folder in directories if (folder / files[0]).exists()), None)
    if directory is None:
        directory = Path('/usr/share/fonts/truetype/dejavu')
        files = ['DejaVuSerif.ttf', 'DejaVuSerif-Bold.ttf', 'DejaVuSerif-Italic.ttf', 'DejaVuSerif-BoldItalic.ttf']
    for alias, filename in zip(['Article', 'ArticleB', 'ArticleI', 'ArticleBI'], files):
        pdfmetrics.registerFont(TTFont(alias, str(directory / filename)))
    pdfmetrics.registerFontFamily('Article', normal='Article', bold='ArticleB', italic='ArticleI', boldItalic='ArticleBI')


class Article:
    def __init__(self, evidence, output):
        register_fonts()
        self.evidence, self.output = Path(evidence), Path(output)
        self.results = {dataset: self.read_input_json(self.evidence / f'{dataset}_results.json')
                        for dataset in ('kion', 'yambda')}
        self.behavior_data = {
            dataset: self.read_input_json(self.evidence / f'{dataset}_behavior.json')
            for dataset in self.results
        }
        self.width = A4[0] - 44 * mm
        self.story = []
        self.styles = {
            'title': ParagraphStyle('title', fontName='ArticleB', fontSize=20, leading=24,
                                   alignment=TA_CENTER, spaceAfter=16),
            'subtitle': ParagraphStyle('subtitle', fontName='Article', fontSize=10, leading=13,
                                      alignment=TA_CENTER, spaceAfter=17),
            'body': ParagraphStyle('body', fontName='Article', fontSize=10.6, leading=13.8,
                                  alignment=TA_JUSTIFY, spaceAfter=8),
            'abstract': ParagraphStyle('abstract', fontName='Article', fontSize=10.1, leading=13.2,
                                      alignment=TA_JUSTIFY, leftIndent=11, rightIndent=11, spaceAfter=14),
            'heading': ParagraphStyle('heading', fontName='ArticleB', fontSize=13.5, leading=17,
                                     spaceBefore=6, spaceAfter=10, keepWithNext=True),
            'small': ParagraphStyle('small', fontName='Article', fontSize=9.1, leading=11.6, spaceAfter=8),
            'cell': ParagraphStyle('cell', fontName='Article', fontSize=8.3, leading=10.5),
            'head': ParagraphStyle('head', fontName='ArticleB', fontSize=8.3, leading=10.5),
            'equation': ParagraphStyle('equation', fontName='ArticleI', fontSize=10.4, leading=17,
                                      alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
        }


    def read_input_json(self, path):
        return json.loads(Path(path).read_text())


    def p(self, text, style='body'):
        self.story.append(Paragraph(text, self.styles[style]))

    def h(self, text):
        self.p(text, 'heading')

    def page(self):
        while self.story and isinstance(self.story[-1], Spacer):
            self.story.pop()
        self.story.append(PageBreak())

    def table(self, headers, rows, widths, caption):
        self.p(caption, 'small')
        cells = [[Paragraph(str(value), self.styles['head']) for value in headers]]
        cells += [[Paragraph(str(value), self.styles['cell']) for value in row] for row in rows]
        table = Table(cells, colWidths=[self.width * width for width in widths], repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LINEABOVE', (0, 0), (-1, 0), .8, colors.black),
            ('LINEBELOW', (0, 0), (-1, 0), .5, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), .8, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(.966, .966, .966)]),
        ]))
        self.story.extend([table, Spacer(1, 7)])

    def figure(self, filename, height, caption):
        path = self.evidence / 'figures' / filename
        self.story.append(Image(str(path), width=self.width, height=height, kind='proportional'))
        self.p(caption, 'small')

    def title(self):
        kion, yambda = self.results['kion'], self.results['yambda']
        self.p('RL-дообучение последовательных<br/>рекомендаций на KION и Yambda', 'title')
        self.p('GPT-2, модель отклика и RL на KION и Yambda<br/>Временная валидация, независимые запуски и позднее подтверждение', 'subtitle')
        self.p('<b>Аннотация.</b> Проверяется, улучшает ли RL по предсказанной доле просмотра или прослушивания '
               'последовательные рекомендации GPT-2. На KION и Yambda сравниваются три независимых обучения GPT-2, '
               'RLOO с ансамблем CatBoost, контроль без градиента награды, ItemKNN и переранжирование. '
               'Все рецепты выбираются по валидации, затем оцениваются на двух поздних периодах. '
               f'Без регуляризации рост модельной награды на {pct(relative(kion, "raw_h5", "predicted_reward@10"))} '
               f'и {pct(relative(yambda, "raw_h5", "predicted_reward@10"))} сопровождается потерей NDCG@10 '
               f'{pct(-relative(kion, "raw_h5", "ndcg@10"))} и {pct(-relative(yambda, "raw_h5", "ndcg@10"))} соответственно. '
               'Этот эффект повторяется на подтверждении. Регуляризация удерживает точность ближе к исходной, '
               'но добавочное улучшение completion-weighted Hit по сравнению с согласованным контролем не подтверждено. '
               'Выбранный RL проходит границу допустимой потери NDCG 5% на KION, но не на основном тесте Yambda. '
               'В обоих наборах общий валидационный выбор - ItemKNN. Изменение награды, концентрация выдачи и наблюдаемое '
               'время потребления дают разные выводы о качестве. Офлайн-результаты не устанавливают причинное влияние '
               'рекомендаций на удовлетворённость или возвращаемость.', 'abstract')
        self.h('1. Исследовательский вопрос')
        self.p('У next-item prediction есть ясная цель: восстановить следующий объект по истории пользователя. При RL цель '
               'можно заменить, например, на ожидаемую долю досмотра. Но модель получает возможность улучшать именно оценку '
               'критика: выбирать удобные ему объекты, сужать каталог или превращать собственные рекомендации в мнимую историю '
               'потребления. Поэтому рост награды требует отдельной проверки релевантности и поведения выдачи.')
        self.p('Здесь проверяются четыре вопроса. Даёт ли RL что-либо сверх простого переранжирования тем же критиком? '
               'Какую часть эффекта объясняет продолжение обычного обучения? Что меняет горизонт в один и пять шагов? '
               'Позволяют ли доступные данные связать награду с длительным интересом пользователя? Положительный результат '
               'определяется по отложенным наблюдениям, а не по максимальной оценке обученной reward model.')

    def related(self):
        self.page()
        self.h('2. Связь с предшествующими работами')
        self.p('Petrov и Macdonald [1] используют GPT-2 для последовательной генерации списка рекомендаций, а затем PPO '
               'для согласования списка с точностью и дополнительными целями. В их схеме есть обучение по выдаче модели-учителя '
               'и обучаемая value model. В нашей работе исходная модель учится непосредственно по наблюдаемым next-item целям, '
               'а RL использует leave-one-out baseline. Это самостоятельный эксперимент с другой наградой и протоколом оценки.')
        self.p('Matveev и соавторы [2] рассматривают многошаговую оптимизацию двухбашенного retrieval на Yambda, моделируют '
               'отклик и используют поправки для off-policy обучения и оценки. Из этой работы заимствован принцип: органические '
               'прослушивания входят в состояние, а рекомендованные прослушивания служат целями. Наша политика семплирует '
               'новые действия из текущего распределения и оптимизируется в фиксированной модели награды; это не воспроизведение '
               'их off-policy алгоритма или doubly robust оценки. Настоящие вероятности показа в открытых данных отсутствуют.')
        self.p('В прикладном разборе Т-Банка [3] описаны SASRec с RL-компонентами и трудности переноса офлайн-прироста '
               'в бизнес-эффект: увеличение числа кликов не сопровождалось ростом объёма кэшбэка. Отдельно отмечено, что '
               'последующий бустинг может изменить эффект RL-модели. Этот опыт мотивирует наши сравнения с простым ранжированием '
               'и независимую оценку результата. Закрытая архитектура банка и его A/B-тест в этой работе не воспроизводятся.')
        self.p('REINFORCE Leave-One-Out (RLOO) выбран как простой метод без отдельной value network; его применение '
               'к обучению по модельной награде обсуждается Ahmadian и соавторами [4]. Для каждого исходного контекста '
               'генерируются несколько траекторий, а награды других траекторий служат baseline. Это уменьшает вариативность '
               'градиента, но не исправляет систематические ошибки самой reward model.')
        self.h('Что изменилось относительно первого эксперимента')
        self.p('Первый вариант исследования использовал KION, один общий SFT checkpoint и три RL seed. Он показал, что '
               'увеличение награды может сопровождаться падением точности. В расширенном исследовании прежний тест 2-8 августа '
               'переведён в development, увеличены обучающая выборка и каталог, добавлены независимые SFT-запуски, согласованный '
               'контроль, переранжирование и репликация на потоке Yambda. Ранее просмотренные результаты не выдаются за новый тест.')
        self.p('Для KION период 9-15 августа впервые используется для ранжирования, но активность до 15 августа уже участвовала '
               'в прежнем анализе семидневного возврата. Поэтому 16-22 августа выделены как отдельное, более строгое подтверждение. '
               'Все числа старого варианта сохранены в архиве; выводы этой статьи опираются на расширенный протокол.')

    def data(self):
        self.page()
        self.h('3. Данные и границы применимости')
        self.p('<b>KION.</b> Данные содержат агрегированные пары user-item с датой последнего просмотра [5, 6]. '
               'Выбраны 30 000 пользователей по истории до границы train; каталог содержит 5000 наиболее частых объектов train. '
               'При точности времени в один день все цели пользователя за этот день имеют общий префикс из более ранних дней. '
               'Нельзя восстановить порядок просмотров внутри дня; старые дни упорядочены по item ID.')
        self.p('Агрегация повторных просмотров к последней записи ограничивает даже хронологическое разбиение: в истории '
               'может отсутствовать более ранний просмотр той же пары. Доля просмотра также относится к агрегированной записи. '
               'Таким образом, KION позволяет проверить поведение модели на доступном префиксе выгрузки, но не восстановить '
               'точный журнал, доступный сервису в момент показа.')
        self.p('<b>Yambda.</b> Использована официальная минимальная версия multi-event, обозначенная 50M [7, 8]. В файле '
               '47 790 449 событий 10 000 пользователей, включая 46 467 212 прослушиваний. Для опыта выбраны 2000 пользователей, '
               'пригодных по train, и 10 000 треков по частоте в train. Префикс содержит только более ранние прослушивания, '
               'включая органические. Цели - прослушивания из рекомендаций. Лайки и другие явные реакции не превращаются '
               'в дополнительные прослушивания. Метки времени выражены в относительных секундах с точностью пять секунд.')
        rows = []
        labels = {'train': 'Обучение', 'rm_valid': 'Выбор GPT-2 / RM', 'valid': 'Выбор RL / rerank',
                  'test': 'Основной тест', 'confirmation': 'Подтверждение'}
        for split in labels:
            cells = [labels[split]]
            for dataset in ('kion', 'yambda'):
                manifest = self.results[dataset]['data_manifest']
                cells += [f"{manifest['rows'][split]:,}".replace(',', ' '), pct(manifest['unknown_target_fraction'][split])]
            rows.append(cells)
        self.table(['Период', 'KION, строк', 'Вне каталога', 'Yambda, строк', 'Вне каталога'], rows,
                   [.28, .18, .16, .21, .17], 'Таблица 1. Объём используемых примеров. Неизвестные цели остаются промахами.')
        self.p('В Yambda равномерно выбирается не более 100 обучающих и 20 оценочных целей на пользователя в каждом периоде. '
               'Выбор не зависит от награды. Основные метрики сначала усредняются по целям пользователя, затем по пользователям. '
               'Это контролируемая выборка событий из малого набора, а не обучение на всех 50 миллионах событий.')
        catalog_path = self.evidence / 'catalog.json'
        if catalog_path.exists():
            catalog = self.read_input_json(catalog_path)['datasets']
            parts = []
            for dataset in ('kion', 'yambda'):
                shares = catalog[dataset]['splits']['test']['fractions_micro']
                parts.append(f"{dataset.upper()}: {pct(shares['train_seen_excluded'])} всех тестовых целей уже встречались "
                             f"в train когорты, но не вошли в каталог; {pct(shares['unseen_in_cohort_train'])} не встречались в её train")
            self.p('. '.join(parts) + '. На Yambda две трети OOV относятся к первой категории. Поэтому OOV нельзя '
                   'целиком объяснить новыми релизами. Отсутствие в train когорты также не устанавливает новизну для сервиса.')
        self.p('У источника Yambda есть отбор пользователей по активности на длинном интервале. Это ограничивает анализ '
               'возвращаемости: полученная когорта заранее благополучнее общей аудитории. Статические связи треков с исполнителями '
               'используются как метаданные; исторических снимков этих связей нет.')

    def protocol(self):
        self.page()
        self.h('4. Временной протокол и оценка')
        self.table(['Назначение', 'KION, 2021 год', 'Yambda, относительные секунды'], [
            ['Train', 'До 26 июля', 't &lt; 17 656 610'],
            ['Выбор GPT-2 и RM', '26 июля - 1 августа', '17 656 610 ≤ t &lt; 19 983 235'],
            ['Выбор рецепта', '2-8 августа', '19 983 235 ≤ t &lt; 22 115 765'],
            ['Основной тест', '9-15 августа', '22 115 765 ≤ t &lt; 24 126 675'],
            ['Подтверждение', '16-22 августа', '24 126 675 ≤ t &lt; 26 000 005'],
        ], [.25, .30, .45], 'Таблица 2. Последовательные непересекающиеся интервалы. Границы Yambda - квантили времени 60/70/80/90%, без использования откликов.')
        self.p('В каждом контексте ранжируется весь замороженный каталог. Для KION исключены все известные предыдущие '
               'просмотры; для музыки знакомые треки разрешены. На тесте Yambda повторно встречавшиеся треки составляют '
               f"{pct(self.results['yambda']['data_manifest']['repeated_target_fraction']['test'])} целей, поэтому запрет повторов "
               'менял бы саму задачу. При генерации списка дубли сгенерированных объектов запрещены в обоих наборах.')
        self.p('Top-10 для основных метрик получается одним ранжированием по реальной истории. Авторегрессионная '
               'генерация пяти объектов исследуется отдельно; её модельная награда не подменяет оценку первого шага.')
        self.p('Основной показатель полезности - user-macro completion-weighted Hit@10; NDCG@10 служит ограничением '
               'релевантности и метрикой non-inferiority. Recall@10 описывает частоту совпадений; при одной цели '
               'он совпадает с HitRate. Фактический completion-weighted Hit@10 равен наблюдаемой доле '
               'досмотра/дослушивания цели, умноженной на '
               'индикатор её попадания в top-10. Она учитывает долю потребления совпавшей цели, а не только факт следующего '
               'события. Это характеристика журнала старой политики, а не ценность нового показа.')
        self.p('Checkpoint GPT-2 и параметры RM выбираются на первом валидационном интервале. На втором выбирается единый '
               'для трёх seed рецепт: максимальный completion-weighted Hit@10 при среднем NDCG не ниже 95% GPT-2. '
               'Исходная модель всегда допустима. Для переранжирования тем же правилом выбираются тип награды и коэффициент. '
               'Затем сохраняется frozen_selection.json; тест и подтверждение не меняют выбор.')
        self.p('Парные интервалы получены 2000 bootstrap-выборками пользователей целиком. Во всех seed используются одни '
               'и те же пользовательские выборки. Основной интервал условен на обученных checkpoints; разброс обучения '
               'показывается отдельно как SD трёх запусков. Совместный bootstrap пользователей и seed служит дополнительной '
               'проверкой: трёх seed недостаточно для надёжного описания всего распределения обучения.')
        self.p('Незначимое различие не доказывает сохранение точности. Для выбранного RL отдельно проверяется '
               'non-inferiority с относительной границей 5%: нижняя односторонняя 95%-я граница для '
               'NDCG<sub>RL</sub> - 0,95 NDCG<sub>GPT-2</sub> должна быть положительной. В bootstrap пересэмплируются '
               'оба слагаемых. Двусторонние p-value парного рандомизационного теста скорректированы методом Holm '
               'внутри каждого набора и периода. Sign-flip предполагает обменность знаков парных пользовательских разностей; '
               'это не рандомизация показов. Поправка относится к двусторонним тестам Δ=0; границы non-inferiority '
               'не скорректированы на множественность. Это условная проверка офлайн-различий, без причинной интерпретации.')

    def models(self):
        self.page()
        self.h('5. Исходная модель и модель отклика')
        selected = [f"{dataset.upper()}: {self.results[dataset]['sft_selection']['selected']}" for dataset in self.results]
        self.p('<b>GPT-2.</b> Два decoder-слоя и четыре attention heads, item ID в качестве токенов. Для KION на первом '
               'заранее заданном seed сравниваются hidden size 96/128/192 и контекст 32/48/48; для Yambda фиксирован размер 128 '
               'и контекст 48. Выбор: ' + '; '.join(selected) + '. Выбранная архитектура независимо обучается на seed 11, 22, 33. '
               'Контекст включает BOS: 48 токенов вмещают 47 предыдущих объектов. Полная история KION сохраняется '
               'для маски просмотренного. Используются AdamW, dropout 0,1, label smoothing 0,02 по допустимым объектам, warmup 8% и cosine decay. '
               'Лучший checkpoint определяется по user-macro NDCG@10, с early stopping.')
        self.p('<b>Reward model.</b> Цель r = clip(completion / 100, 0, 1). На ранних днях train замораживаются сглаженные '
               'средние отклика объектов, частоты и 24-мерные коллаборативные признаки бинарной матрицы user-item. '
               'На более поздних train-днях обучается ансамбль трёх CatBoost. Признаки описывают объект, жанры или исполнителей, '
               'раннюю статистику, распределение интересов в истории, недавние взаимодействия и соответствие кандидата истории. '
               'В Yambda добавлены длительность и признак предыдущего прослушивания. Текущий отклик цели не является признаком.')
        self.p('Для CatBoost выбирается глубина 4, 6 или 8 и число деревьев до 300/500 с early stopping; критерий выбора '
               '- обычный RMSE на RM-validation. При обучении каждый пользователь имеет одинаковый суммарный вес. '
               'Дополнительно отчёт содержит user-macro ошибки, калибровку и интервалы. Один фиксированный ансамбль RM '
               'общий для трёх policy seed; разброс политик не включает повторное обучение критика. Item-only и history-only абляции '
               'используют ёмкость выбранной полной модели; независимо оптимальными моделями этих классов они не считаются.')
        self.p('Дополнительный критик обучается на непересекающейся трети пользователей позднего train. Он разделяет '
               'ранние статистики и смещения источника с основным ансамблем. Согласие двух критиков помогает обнаружить '
               'часть ошибок оптимизации, но не заменяет внешнюю оценку. Стандартное отклонение ансамбля трактуется '
               'как несогласие моделей; его связь с реальной ошибкой измеряется отдельно.')
        self.p('<b>Классические сравнения.</b> Популярность использует только train. ItemKNN объединяет обучающие цели '
               'и объекты их префиксов в бинарную матрицу взаимодействий. Косинусное сходство учитывает обратный логарифм '
               'активности пользователя и shrinkage; сохраняются 100 соседей объекта. В запросе применяются IDF '
               'и экспоненциальное затухание по позиции. Плотная матрица catalog × catalog не материализуется. '
               'KNN берёт до 64 последних известных объектов, отбрасывая UNK; GPT-2 - последние 47 токенов вместе с UNK. '
               'RM видит всю доступную историю KION или до 48 прослушиваний Yambda. Межсемейные сравнения различаются '
               'и бюджетом памяти; прямые контрасты RL и контроля согласованы по истории.')
        self.p('<b>Переранжирование без обучения.</b> Из top-200 исходного GPT-2 выбирается top-10 по '
               'logit<sub>GPT-2</sub>(a|h) + α r̂(h,a), где α выбирается из {0; 0,25; 0,5; 1; 2; 4; 8}. '
               'На валидации сравниваются r̂ и r̂ - σ. При α=0 восстанавливается исходная выдача. '
               'Будущие положительные объекты не вставляются в кандидаты. Этот baseline проверяет, требуется ли изменение '
               'весов политики для наблюдаемого компромисса.')

    def rl(self):
        self.page()
        self.h('6. RL и согласованные сравнения')
        self.p('Из train равномерно выбирается пользователь, затем его наблюдаемый контекст и известная цель для CE. '
               'Для одного контекста семплируются четыре независимые траектории длины H. Политика добавляет сгенерированный '
               'объект в контекст. В основном варианте критик оценивает каждый кандидат по исходной реальной истории; '
               'вариант appended отдельно проверяет эффект добавления искусственных взаимодействий и в reward model.')
        self.p('G<sub>t</sub><super>(i)</super> = Σ<sub>u=t</sub><super>H-1</super> r̂<sub>u</sub><super>(i)</super>, '
               'A<sub>t</sub><super>(i)</super> = G<sub>t</sub><super>(i)</super> - mean<sub>j≠i</sub> G<sub>t</sub><super>(j)</super>.', 'equation')
        self.p('L = -w<sub>PG</sub> mean[A<sub>t</sub> log π(a<sub>t</sub>|h<sub>t</sub>)] '
               '+ β KL(π || π<sub>0</sub>) + λ L<sub>CE</sub>.', 'equation')
        self.p('Дисконт γ=1. Награды не дифференцируются. KL вычисляется точно по допустимому каталогу в посещённых '
               'состояниях; это локальная регуляризация, не полный градиент траекторного KL. CE берётся на реальном '
               'исходном контексте. При RL dropout выключен и у политики, и у замороженного GPT-2, чтобы случайные '
               'маски не вносили отдельное расхождение распределений.')
        self.table(['Вариант', 'H', 'Награда', 'wPG / β / λ'], [
            ['Согласованный контроль', '5', 'Вычисляется, PG выключен', '0 / 0,2 / 0,5'],
            ['Без ограничений', '5', 'r̂', '1 / 0 / 0'],
            ['Основной, один шаг', '1', 'r̂', '1 / 0,2 / 0,5'],
            ['Основной, пять шагов', '5', 'r̂', '1 / 0,2 / 0,5'],
            ['Appended, KION', '5', 'r̂ с дополненной историей', '1 / 0,2 / 0,5'],
            ['Консервативный', '5', 'r̂ - σ - 0,05(1-support)', '1 / 0,2 / 0,5'],
            ['Возвращаемость, KION', '5', '0,65r̂ + 0,35p̂(return7) - σ', '1 / 0,2 / 0,5'],
            ['Вычитание тренда, Yambda', '5', 'r̂ - m(duration) + m', '1 / 0,2 / 0,5'],
            ['Секунды, Yambda', '5', 'min(r̂ × duration, 600) / 300', '1 / 0,2 / 0,5'],
        ], [.32, .06, .39, .23], 'Таблица 3. Заранее заданная сетка. m(duration) - сглаженная средняя доля в раннем train-бине длительности; support - эвристика частоты, не propensity.')
        self.p('Каждый вариант получает 160 обновлений на KION и 120 на Yambda, batch из 24 контекстов, learning rate '
               '0,00015 и clipping нормы градиента 1. Контроль без PG имеет те же семплирование, KL, CE и seed. '
               'Для H=1 и H=5 совпадает число обновлений и исходных контекстов, но H=5 генерирует в пять раз больше действий. '
               'Это сравнение при равном бюджете обновлений, а не при равных вычислениях. Frozen и appended при H=5 согласованы по числу действий.')

    def reward_results(self):
        self.page()
        self.h('7. Насколько полезна reward model')
        rows = []
        critic_names = [('constant', 'Константа'), ('early_item_mean', 'Раннее среднее объекта'),
                        ('item_only_model', 'Только объект'), ('history_only_model', 'Только история'),
                        ('ensemble', 'Полная RM'), ('audit', 'Дополнительный критик')]
        for dataset, result in self.results.items():
            rm = result['reward_metrics']['test']
            for key, label in critic_names:
                row = rm[key]
                rows.append([dataset.upper() if key == 'constant' else '', label,
                             number(row['rmse']), number(row['mae']), number(row['r2'])])
        self.table(['Данные', 'Модель', 'RMSE', 'MAE', 'R²'], rows, [.14, .38, .16, .16, .16],
                   'Таблица 4. Micro-ошибки по известным фактически потреблённым парам основного теста. Все модели внутри набора оценены на одинаковых парах; user-macro ошибки сохранены отдельно.')
        for dataset, result in self.results.items():
            rm = result['reward_metrics']['test']
            improvement = 1 - rm['ensemble']['rmse'] / rm['early_item_mean']['rmse']
            disagreement = rm['ensemble_disagreement_diagnostics']
            ablation = ('Полная RM не улучшила micro-RMSE относительно item-only.' if dataset == 'kion' else
                        'Item-only слабее полной RM; выигрыш к history-only составляет лишь ' +
                        pct(1-rm['ensemble']['rmse']/rm['history_only_model']['rmse'], 2) + ' RMSE.')
            self.p(f"<b>{dataset.upper()}.</b> Изменение RMSE к раннему среднему объекта: {pct(-improvement, signed=True)}. "
                   f"Корреляция σ с абсолютной ошибкой: {number(disagreement['std_vs_absolute_error_pearson'], 3)}; "
                   f"покрытие отклика интервалом r̂ ± 1,96σ: {pct(disagreement['naive_1_96std_coverage'])}. " + ablation, 'small')
        self.figure('reward_calibration.png', 160,
                    'Рисунок 1. Средняя предсказанная и фактическая доля по бинам прогноза на основном тесте. Размер точки отражает число наблюдений; редкие крайние бины не подтверждают качество на произвольных кандидатах.')
        self.p('Несогласие ансамбля не является калиброванной неопределённостью отклика: вычитание σ остаётся эвристикой. '
               'Критик предсказывает долю при потреблении, а ценность показа зависит ещё и от вероятности выбора объекта. '
               'Без полного журнала показов эта вероятность не оценивается. KL, CE и исходный logit сохраняют сигнал '
               'релевантности, но не калибруют его. Ошибки на потреблённых парах также не характеризуют все действия RL.')

    def dataset_results(self, dataset, section):
        self.page()
        result = self.results[dataset]
        selected = result['selection']['selected_rl']
        digits = 5 if dataset == 'yambda' else 4
        ci_digits = 6 if dataset == 'yambda' else 5
        self.h(f'{section}. Результаты на {dataset.upper()}')
        rows = []
        for family in NAMES:
            if family not in groups(result):
                continue
            label = NAMES[family] + (' *' if family == selected else '')
            rows.append([label, mean_sd(result, family, 'recall@10', digits=digits), mean_sd(result, family, 'ndcg@10', digits=digits),
                         mean_sd(result, family, 'factual_completion_weighted_hit@10', digits=digits),
                         number(mean(result, family, 'predicted_reward@10')) + ' / ' + number(mean(result, family, 'audit_reward@10'))])
        self.table(['Политика', 'Recall@10', 'NDCG@10', 'r × Hit@10', 'RM / критик'], rows,
                   [.31, .17, .17, .18, .17],
                   f'Таблица {5 if dataset == "kion" else 7}. Основной тест: среднее ± SD трёх запусков при общей RM и рецепте; популярность и ItemKNN - одна оценка; RM / дополнительный критик - средние. * - RL, выбранный по валидации.')
        rr = result['selection']['reranker']
        self.p(f"По валидации выбран RL-рецепт <b>{NAMES.get(selected, 'нет допустимого')}</b>; "
               f"для переранжирования α={rr['alpha']}, награда {escape(rr['reward_type'])}. "
               f"Общий выбор среди всех допустимых семейств: <b>{NAMES[result['selection']['selected_overall']]}</b>. "
               'Выбор не пересматривается по числам в этой таблице.')
        if selected:
            text = []
            for split, label in [('test', 'Основной тест'), ('confirmation', 'Подтверждение')]:
                stat = result['statistics'][split][f'{selected}_vs_baseline']['ndcg@10']
                verdict = 'пройдена' if stat['noninferiority']['passes_user'] else 'не пройдена'
                text.append(f"{label}: ΔNDCG {pct(stat['relative_delta'], signed=True)}, "
                            f"абсолютная разность {number(stat['delta'], ci_digits)} "
                            f"[95% CI: {number(stat['ci_user']['low'], ci_digits)}; {number(stat['ci_user']['high'], ci_digits)}]; "
                            f"NI {verdict}, нижняя граница {number(stat['noninferiority']['lower_bound_user'], ci_digits)}")
            self.p('. '.join(text) + '. NI ограничивает относительную потерю 5%; её односторонняя 95%-я граница '
                   'относится к NDCG<sub>RL</sub> - 0,95 NDCG<sub>GPT-2</sub>. Вывод условен на checkpoints.', 'small')
        rows = []
        for family in ['baseline', 'matched_control', 'reranker', selected]:
            if family is None or any(row[0] == NAMES[family] for row in rows):
                continue
            rows.append([NAMES[family], mean_sd(result, family, 'ndcg@10', 'confirmation', digits=digits),
                         mean_sd(result, family, 'factual_completion_weighted_hit@10', 'confirmation', digits=digits),
                         mean_sd(result, family, 'predicted_reward@10', 'confirmation')])
        self.table(['Политика', 'NDCG@10', 'r × Hit@10', 'Награда RM'], rows, [.34, .22, .23, .21],
                   f'Таблица {6 if dataset == "kion" else 8}. Позднее подтверждение для заранее выбранных сравнений.')
        if dataset == 'kion':
            self.p('Набор объектов следующего наблюдаемого дня дополнительно оценивается как множество. '
                   'В знаменателях Recall и идеального DCG сохранены OOV; один день не разбивается на независимые next-item цели.', 'small')
        else:
            self.p('Основные цели включают listen-события с нулевой долей. Срезы с долей ≥ 0,5 и на известной части '
                   'каталога условны на своём составе; они не заменяют основной результат и не исправляют ограниченность каталога.', 'small')
        families = list(dict.fromkeys(['baseline', 'reranker'] + ([selected] if selected else [])))
        secondary = 'set_ndcg@10' if dataset == 'kion' else 'engaged_ndcg@10'
        self.table(['Политика', 'NDCG, известный каталог', 'NDCG, ' + ('набор дня' if dataset == 'kion' else 'доля ≥ 0,5')],
                   [[NAMES[family], mean_sd(result, family, 'known_catalog_ndcg@10'),
                     mean_sd(result, family, secondary)] for family in families], [.38, .31, .31],
                   'Дополнительные срезы основного теста. Знаменатели и состав пользователей отличаются от основной метрики.')

    def tradeoff(self):
        self.page()
        self.h('10. Компромисс награды и релевантности')
        self.figure('frontier.png', 280,
                    'Рисунок 2. Основной тест: точность, фактический completion-weighted Hit и предсказанная награда. Малые точки - отдельные seed; крупные - средние. Это описание заранее заданной сетки, не выбор лучшей тестовой настройки.')
        self.p('Одна координата показывает, насколько выдача похожа на будущие действия старой политики; другая - '
               'что именно мы увеличиваем. Рост предсказанной доли при падении фактического weighted Hit указывает '
               'на расхождение цели критика и доступной внешней проверки. Даже рост обеих координат остаётся '
               'наблюдательным результатом: реакции на объекты, которых нет в журнале, неизвестны.')
        rows = []
        for dataset, result in self.results.items():
            if not result['selection']['selected_rl']:
                continue
            for comparator in ['matched_control', 'reranker']:
                stat = result['statistics']['test'][f'selected_rl_vs_{comparator}']['factual_completion_weighted_hit@10']
                interval = stat['ci_user']
                digits = 6 if dataset == 'yambda' else 5
                rows.append([dataset.upper(), NAMES[comparator], number(stat['delta'], digits),
                             f"[{number(interval['low'], digits)}; {number(interval['high'], digits)}]",
                             pvalue(stat['holm_adjusted_p_two_sided'])])
        self.table(['Данные', 'Сравнение выбранного RL с', 'Δ r × Hit', '95% CI', 'p Holm'], rows,
                   [.13, .33, .16, .23, .15],
                   'Таблица 9. Парные сравнения, условные на checkpoints. Интервалы - кластерный bootstrap пользователей; p-value скорректированы по полной семье сравнений периода.')
        self.p('Вклад PG при одинаковых H=5, KL/CE и бюджете обновлений изолирует прямой контраст anchored H5 '
               'с контролем без PG в разделе 13. Если выбран другой рецепт, его сравнение с этим контролем относится '
               'к процедуре целиком: могут различаться также горизонт или регуляризация. Разница с переранжированием '
               'отвечает на более прикладной вопрос: оправдано ли '
               'переобучение политики, если тот же критик можно применить только при выдаче. Стоимость инференса '
               'переранжирования выше: дополнительно оцениваются 200 кандидатов на контекст.')

    def behavior(self):
        self.page()
        self.h('11. Персонализация, повторы и длительность')
        rows = []
        for dataset, result in self.results.items():
            for family in ['baseline', 'raw_h5', 'matched_control', 'reranker', result['selection']['selected_rl']]:
                if family is None or any(row[0] == dataset.upper() and row[1] == NAMES[family] for row in rows):
                    continue
                rows.append([dataset.upper(), NAMES[family], pct(mean(result, family, 'catalog_coverage@10')),
                             pct(mean(result, family, 'top1_concentration')),
                             pct(self.behavior_data[dataset]['families'][family]['test']['users_majority_modal_top1_fraction']['mean'])])
        self.table(['Данные', 'Политика', 'Покрытие', 'Частый top-1', 'Пользователи &gt;½'], rows,
                   [.13, .34, .16, .18, .19],
                   'Таблица 10. Средние трёх seed. Для каждого запуска найден глобально наиболее частый первый объект. Показаны доля различных контекстов с ним и доля пользователей, получающих его более чем в половине контекстов. Последний описательный срез добавлен после основного анализа; он не участвовал в выборе модели.')
        for dataset, result in self.results.items():
            behavior = self.behavior_data[dataset]
            selected = result['selection']['selected_rl']
            families = list(dict.fromkeys(family for family in ['baseline', 'raw_h5', selected, 'reranker'] if family))
            parts = []
            for family in families:
                label = {'baseline': 'GPT-2', 'raw_h5': 'Raw RL', 'reranker': 'Rerank'}.get(family, 'RL*')
                if family == selected and family in ('raw_h5', 'reranker'):
                    label += '*'
                values = behavior['families'][family]['test']
                overlap = pct(values['top10_overlap_with_baseline']['mean'])
                if dataset == 'yambda':
                    familiar = pct(values['familiar_top10_share']['mean'])
                    parts.append(f'{label}: {familiar} / {overlap}')
                else:
                    parts.append(f'{label}: {overlap}')
            if dataset == 'yambda':
                self.p('<b>Yambda, знакомые объекты / overlap с GPT-2:</b> ' + '; '.join(parts) + '. '
                       'Знакомость проверяется по последним 48 прослушиваниям, с равным весом контекстов; '
                       'overlap - доля пересечения top-10, сначала по контекстам пользователя, затем по пользователям. '
                       'Приведены средние по seed, а не доли знакомых объектов за всю жизнь пользователя.', 'small')
            else:
                self.p('<b>KION, overlap с GPT-2:</b> ' + '; '.join(parts) + '. '
                       'Знакомые объекты здесь исключены самим протоколом; отсутствие повторов не является эффектом RL.', 'small')
        self.figure('duration.png', 185,
                    'Рисунок 3. Yambda: средняя длительность рекомендованных треков и ожидаемые секунды для заранее заданных вариантов. Используется длительность, замороженная по train; ожидаемые секунды - произведение доли RM на длительность, без повторного проигрывания.')
        self.p('Длительности треков позволяют проверить сдвиг к короткому контенту. Секундная награда и вычитание '
               'раннего тренда задают разные предпочтения, но не устанавливают влияние длины на удовлетворённость. '
               'Вычитание тренда - эвристика: остаточная связь награды с длиной может сохраняться.', 'small')
        self.p('Для KION total_dur - агрегированное потребление, а не длина объекта; такой признак кандидата '
               'принёс бы отклик в признаки. Доля сериалов характеризует состав контента, не длительность. '
               'Названия и частоты объектов сохранены в таблицах поведения.', 'small')

    def longterm(self):
        self.page()
        self.h('12. Многошаговая генерация и возвращаемость')
        self.figure('rollout.png', 210,
                    'Рисунок 4. Greedy-генерация пяти объектов на общей выборке до 256 различных контекстов, seed=42. Средние вычислены по контекстам и policy seed, не user-macro. Показаны оценка искусственной истории и её сдвиг относительно реальной; ответы пользователя не наблюдаются.')
        self.p('Контекстный сдвиг сравнивает две оценки одной траектории без наблюдаемого потребления. '
               'Unmasked разрешает сгенерированные дубли и просмотренное KION; внутренние маски reranker сохраняются.', 'small')
        self.p('В Yambda исходная история RM содержит не более 48 объектов, а на пятом оцениваемом шаге может '
               'содержать уже 52. Поэтому часть контекстов выходит за обучающий диапазон длины. Это возможный '
               'источник сдвига оценки, а не установленное объяснение наблюдаемого эффекта.', 'small')
        self.p('Frozen-RM оптимизирует список по исходному состоянию. Модели перехода интересов или остановки '
               'нет: H=5 означает пять генераций, не доказанное планирование клиентского пути. Appended меняет '
               'оценку при добавлении гипотетических взаимодействий.')
        rows = []
        for dataset, result in self.results.items():
            values = result['reward_metrics']['test'].get('return7', {})
            for key, label in [('constant', 'Константа'), ('history_only', 'Только история'), ('full', 'История + кандидат')]:
                model = values.get(key, {})
                rows.append([dataset.upper() if key == 'constant' else '', label,
                             number(model.get('brier'), 6), number(model.get('auc'), 3)])
        self.table(['Данные', 'Предсказание возврата', 'Brier', 'AUC'], rows, [.16, .44, .20, .20],
                   'Таблица 11. Возврат в следующий активный день в течение семи суток. Полная и history-only модели обучены на одинаковых строках и весах с одинаковой ёмкостью.')
        counts = []
        for dataset, result in self.results.items():
            ret = result['reward_metrics']['test'].get('return7', {})
            counts.append(f"{dataset.upper()}: {ret.get('n_user_days', 0)} дозревших user-days, доля возврата {pct(ret.get('label_mean'))}")
        self.p('; '.join(counts) + '. Веса - равный суммарный вес каждого user-day.', 'small')
        self.p('На KION добавление кандидата не улучшило Brier относительно history-only; на Yambda улучшение '
               'меньше 0,0001 в обоих дозревших периодах, а AUC полной модели ниже history-only. Эти результаты '
               'не дают убедительного основания использовать прогноз возврата как долгосрочную награду за выбор конкретного объекта.', 'small')
        self.p('return7 - активность в другие условные сутки в течение семи дней. Окна полностью дозрели до '
               'train cutoff либо конца источника; на подтверждении KION пригодных меток нет. '
               'Это прогноз привычной активности, а не причинный эффект объекта.', 'small')

    def conclusions(self):
        self.page()
        self.h('14. Что позволяет заключить эксперимент')
        for dataset, result in self.results.items():
            selected = result['selection']['selected_rl']
            if selected is None:
                self.p(f'<b>{dataset.upper()}.</b> Ни один RL-рецепт не прошёл валидационный порог точности. '
                       'Выбранная процедура не даёт основания предпочесть RL исходной политике.')
                continue
            self.p(f"<b>{dataset.upper()}.</b> Выбранный {NAMES[selected]} меняет NDCG@10 относительно GPT-2 на "
                   f"{pct(relative(result, selected, 'ndcg@10'), signed=True)} в основном тесте и "
                   f"{pct(relative(result, selected, 'ndcg@10', 'confirmation'), signed=True)} на подтверждении; "
                   f"completion-weighted Hit@10 - на {pct(relative(result, selected, 'factual_completion_weighted_hit@10'), signed=True)} "
                   f"и {pct(relative(result, selected, 'factual_completion_weighted_hit@10', 'confirmation'), signed=True)}. " +
                   ('NI проходит в обоих периодах, в том числе при совместном bootstrap seed и пользователей. '
                    'Это сохранение потери внутри границы 5%, а не отсутствие потери: поздний NDCG снижается и после Holm-поправки.'
                    if dataset == 'kion' else
                    'В основном тесте NI не проходит. На подтверждении она проходит только условно на checkpoints, '
                    'но не при совместном bootstrap. Устойчивое сохранение точности этим не установлено.'))
        self.p('В согласованном контрасте H5 с контролем без PG награда критика растёт в обоих наборах и периодах, '
               'но выигрыш фактического weighted Hit не подтверждён. На KION PG дополнительно снижает NDCG '
               '(Holm p=0,024 / 0,006). Пять шагов, дополнение истории RM и награда за возврат также не дали '
               'подтверждённого преимущества. Общий валидационный выбор в обоих наборах - ItemKNN; '
               'переранжирование остаётся сильной альтернативой без RL. Его положительные изменения weighted Hit '
               'не проходят общую Holm-семью сравнений с GPT-2 на уровне 0,05.')
        yambda = self.results['yambda']
        user_share = self.behavior_data['yambda']['families']['raw_h5']['test']['users_majority_modal_top1_fraction']['mean']
        self.p('Схлопывание выдачи имеет разный состав. В KION raw RL сдвигается к более популярным объектам; '
               f"в Yambda новизна по train-частоте (-log<sub>2</sub> сглаженной вероятности объекта), напротив, растёт с {number(mean(yambda, 'baseline', 'novelty_bits@10'), 2)} "
               f"до {number(mean(yambda, 'raw_h5', 'novelty_bits@10'), 2)} бит. При этом один и тот же трек "
               f"становится первым более чем в половине контекстов у {pct(user_share)} пользователей в среднем по seed. "
               'Высокое разнообразие исполнителей внутри списка не исключает одинаковой выдачи разным людям. '
               'Рост обоих связанных критиков при падении попаданий согласуется с несоответствием цели: '
               'доля при потреблении не учитывает вероятность выбрать объект. Это объяснение согласуется с данными, '
               'но не отделено причинно от ошибок критиков вне обучающей поддержки.')
        self.h('Ограничения')
        self.p('Оба набора наблюдательные. Нет полного лога показов и известных propensities, поэтому отсутствует '
               'идентифицируемая off-policy оценка причинной ценности. Критик обучен на потреблённых объектах; '
               'неопределённость вне этой поддержки не калибрована. KION агрегирован и имеет грубое время; '
               'Yambda содержит отбор по активности, малую пользовательскую подвыборку и существенный дрейф каталога. '
               'Сравнение известных объектов - только дополнительный срез. Три seed и одна общая RM не покрывают '
               'неопределённость переобучения критика и выбора процедуры. Бюджеты истории у межсемейных baseline различаются.')
        self.h('Что проверять дальше')
        self.p('Для практического продолжения нужны журнал кандидатов и показов с вероятностями, расширение каталога '
               'без подстановки будущих позитивов и проверка RM на рандомизированных показах. Многошаговая оптимизация '
               'потребует модели отклика и остановки сессии, проверенной на реальных переходах. Онлайн-эксперимент '
               'нужен для проверки пользовательской пользы, дозревшего возврата и ограничений на концентрацию. '
               'Представленных результатов недостаточно для утверждения о готовности RL к внедрению.')

    def ablations_and_feedback(self):
        contrast_path = self.evidence / 'paired_ablation_statistics.json'
        feedback_path = self.evidence / 'feedback.json'
        contrast = self.read_input_json(contrast_path)
        feedback = self.read_input_json(feedback_path)
        self.page()
        self.h('13. Прямые абляции и явный отклик')
        self.p('Сравнение каждого варианта с GPT-2 не заменяет прямых контрастов PG, горизонтов и истории RM. Ниже приведены парные '
               'разности на тех же пользователях и SFT seed. План вычисления этих контрастов дописан после запуска GPU, '
               'до просмотра новых тестовых предсказаний; вопросы PG, H1/H5 и frozen/appended уже были заданы в основном протоколе. '
               'Ни обучение, ни выбор рецепта не менялись.')
        rows = []
        for dataset, details in contrast['datasets'].items():
            for entry in details['splits']['test']['comparisons'].values():
                candidate, baseline = entry['candidate_family'], entry['baseline_family']
                labels = {('anchored_h5', 'matched_control'): 'H5: PG - без PG',
                          ('anchored_h5', 'anchored_h1'): 'H5 - H1',
                          ('appended_h5', 'anchored_h5'): 'Appended - frozen'}
                if (candidate, baseline) not in labels:
                    continue
                cells = [dataset.upper(), labels[(candidate, baseline)]]
                for metric in ('ndcg@10', 'factual_completion_weighted_hit@10'):
                    stat = entry['metrics'][metric]
                    digits = 6 if dataset == 'yambda' else 5
                    cells.append(f"{number(stat['delta'], digits)}<br/>[{number(stat['ci_user']['low'], digits)}; {number(stat['ci_user']['high'], digits)}]")
                rows.append(cells)
        self.table(['Данные', 'Прямой контраст', 'ΔNDCG, 95% CI', 'Δ r × Hit, 95% CI'], rows,
                   [.13, .27, .30, .30], 'Таблица 12. Основной тест. Парные user-cluster интервалы условны на checkpoints и не скорректированы на множественность. Для всех прямых контрастов сохранена отдельная Holm-семья из 12 двусторонних тестов на набор и период; полные контрасты наград и подтверждение доступны в материалах.')
        self.p('Явный отклик Yambda сопоставляется с ближайшим строго предшествующим прослушиванием того же user-item '
               'в пределах 24 часов. В сопоставлении участвуют и органические, и несэмплированные прослушивания. '
               'Неоднозначный порядок при одинаковом времени, дубликаты ключа и незавершённые 24-часовые окна исключены. '
               'like24 означает факт события, а не итоговое состояние избранного; unlike и undislike сохраняются отдельно.')
        selected = self.results['yambda']['selection']['selected_rl']
        families = list(dict.fromkeys(['baseline', 'raw_h5', 'reranker'] + ([selected] if selected else [])))
        rows = []
        for split, label in [('test', 'Тест'), ('confirmation', 'Поздний')]:
            for family in families:
                values = feedback['splits'][split]['family_means'][family]
                rows.append([label, NAMES[family], number(values.get('like24_weighted_hit@10'), 5),
                             number(values.get('dislike24_weighted_hit@10'), 5)])
        self.table(['Период', 'Политика', 'like24 × Hit@10', 'dislike24 × Hit@10'], rows,
                   [.13, .39, .24, .24], 'Таблица 13. Вторичные метрики явного отклика: среднее по пользователям и трём policy seed. Это совпадения с событиями журнала, не вероятности отклика на новую выдачу.')
        counts = feedback['label_counts']
        self.p(f"В основном тесте пригодны {counts['test']['eligible_rows']} строк: "
               f"like={counts['test']['positive_labels']['like24']}, dislike={counts['test']['positive_labels']['dislike24']}; "
               f"в позднем периоде - {counts['confirmation']['eligible_rows']} строк, "
               f"like={counts['confirmation']['positive_labels']['like24']}, dislike={counts['confirmation']['positive_labels']['dislike24']}. "
               'Отсутствие лайка не означает негативного отклика. Из-за редкости дизлайков нулевые совпадения или '
               'интервалы около нуля не доказывают безопасность политики. Аудит не используется для выбора модели.', 'small')

    def appendix(self):
        self.page()
        self.h('Материалы и воспроизведение')
        self.p('Короткая инструкция запуска находится в корневом README. Конфиги configs/kion.json и configs/yambda.json '
               'задают сетку экспериментов, временные интервалы и критерий выбора. '
               'Сырые данные скачиваются из официальных источников; веса и пользовательские префиксы в пакет статьи не включены.')
        rows = []
        for dataset, result in self.results.items():
            architecture = result['sft_selection']['architecture_search'][result['sft_selection']['selected']]['architecture']
            cfg = result['config']
            hardware = ('Tesla T4 + CPU' if dataset == 'kion' else 'Tesla T4') + ' / CPU'
            rows.append([dataset.upper(), str(architecture['embedding_dim']), str(architecture['max_length']),
                         str(cfg['sft_epochs']), str(cfg['rl_steps']), hardware])
        self.table(['Данные', 'Hidden', 'Контекст', 'SFT, до эпох', 'RL, шагов', 'Train / оценка'], rows,
                   [.15, .12, .15, .18, .16, .24], 'Таблица 16. Фактически выбранная ёмкость и фиксированные бюджеты. Batch SFT=256; batch RL=24 × 4 траектории; seed RL = seed SFT + 1000.')
        self.p('SFT обоих наборов и RL Yambda обучены на Tesla T4 в приватных Kaggle kernels. Из-за ограничения времени '
               'KION продолжен локально: все семь вариантов seed 33, включая контроль без PG, заново обучены на CPU; '
               'seed 11/22 остались GPU-запусками. Маршрут продолжения зафиксирован до теста по доступности вычислений, '
               'без сравнения качества. Прямые контрасты RL и контроля без PG внутри seed согласованы по устройству, но разброс трёх запусков KION '
               'включает и различие оборудования. Отложенная оценка всех политик '
               'на обоих периодах выполнена заново на Mac CPU исходным кодом: этот выбор принят до просмотра новых '
               'тестовых рекомендаций по замеру скорости на валидации. Частичные GPU-предсказания не смешиваются '
               'с CPU-результатами. Точное побитовое совпадение на другой версии CUDA/PyTorch не заявляется.')
        self.h('Литература')
        references = [
            ('1', 'Petrov A. V., Macdonald C. Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning. 2024.', 'https://arxiv.org/abs/2403.04875'),
            ('2', 'Matveev A. et al. Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction. 2026.', 'https://arxiv.org/abs/2607.02818'),
            ('3', 'Лашинин О. RL RecSys в проде: хайп или игра вдолгую. Т-Банк, 2025.', 'https://habr.com/ru/companies/tbank/articles/882478/'),
            ('4', 'Ahmadian A. et al. Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs. 2024.', 'https://arxiv.org/abs/2402.14740'),
            ('5', 'KION Dataset. Официальный репозиторий.', 'https://github.com/irsafilo/KION_DATASET'),
            ('6', 'Сафило И., Тихонович Д., Петров А. В., Игнатов Д. И. MTC Kion: набор данных для рекомендательных систем на основе неявного отклика, контекстных признаков и анализа последовательностей. 2023.', 'https://doi.org/10.31857/S2686954323600428'),
            ('7', 'Ploshkin A. et al. Yambda-5B - A Large-Scale Multi-modal Dataset for Ranking And Retrieval. 2025.', 'https://arxiv.org/abs/2505.22238'),
            ('8', 'Yandex. Yambda: official dataset card and 50M multi-event data.', 'https://huggingface.co/datasets/yandex/yambda'),
            ('9', 'PyTorch. Reproducibility: ограничения воспроизводимости между платформами и версиями.', 'https://docs.pytorch.org/docs/2.14/notes/randomness.html'),
        ]
        for index, title, url in references:
            self.p(f'[{index}] {title} <link href="{url}" color="#202020"><u>{escape(url.replace("https://", ""))}</u></link>.', 'small')

    def duration_supplement(self):
        path = self.evidence / 'duration_utility.json'
        audit = self.read_input_json(path)
        self.page()
        self.h('Дополнение. Наблюдаемое время и сетка переранжирования')
        self.p('Проверка времени добавлена после обучения и валидации, до анализа новых отложенных предсказаний. '
               'Вторичная метрика равна Hit@10 × min(r × d, 600), где r - наблюдаемая ограниченная доля, '
               'd - длина целевого трека из события в секундах. Она учитывает одно проигрывание с ограничением '
               '600 секунд, а не повторы или причинный эффект новой выдачи. Неизвестные цели остаются нулевыми промахами; '
               'известные цели без пригодной длины исключаются одинаково для всех политик.', 'small')

        def display(value):
            text = number(value['mean'], 3)
            if value.get('n_runs', 1) > 1:
                text += ' ± ' + number(value.get('seed_sd'), 3)
            return text

        families = ['baseline', 'itemknn', 'anchored_h5', 'duration_neutral_h5', 'seconds_h5', 'reranker']
        rows = [[NAMES[family], display(audit['splits']['test']['families'][family]),
                 display(audit['splits']['confirmation']['families'][family])] for family in families]
        self.table(['Политика', 'Тест, с × Hit@10', 'Подтверждение, с × Hit@10'], rows, [.40, .30, .30],
                   'Таблица 14. Ограниченное время одного прослушивания × Hit на целях журнала: user-macro среднее ± SD трёх запусков; ItemKNN - одна оценка. Это не ожидаемое время после показа списка.')
        populations = []
        for split, label in [('test', 'Тест'), ('confirmation', 'Подтверждение')]:
            values = audit['splits'][split]['families']['baseline']
            populations.append(f"{label}: {values['n_eligible_rows']} строк, {values['n_eligible_users']} пользователей")
        self.p('; '.join(populations) + '. Обучение использовало прогноз RM и замороженную медианную длину объекта; '
               'здесь ограничен каждый наблюдаемый отклик до усреднения. Эти две величины не тождественны.', 'small')
        for name, label in [('seconds_h5_vs_anchored_h5', 'Секундная награда - основная H5'),
                            ('duration_neutral_h5_vs_anchored_h5', 'Вычитание тренда - основная H5')]:
            parts = []
            for split, split_label in [('test', 'тест'), ('confirmation', 'подтверждение')]:
                stat = audit['splits'][split]['comparisons'][name]
                parts.append(f"{split_label}: Δ={number(stat['delta'], 3)} с, "
                             f"95% CI [{number(stat['ci_user']['low'], 3)}; {number(stat['ci_user']['high'], 3)}], "
                             f"p Holm={pvalue(stat['holm_adjusted_p_two_sided'])}")
            self.p('<b>' + label + ':</b> ' + '; '.join(parts) + '.', 'small')
        self.p('Интервалы условны на checkpoints и не скорректированы на множественность; Holm-поправка охватывает два контраста '
               'внутри периода, не все проверки статьи. Метрика не меняет выбор рецепта.', 'small')
        self.figure('rerank_validation.png', 216,
                    'Рисунок 5. Только policy-validation: вся сетка α и двух оценок награды, средние трёх seed. Справа от серой границы выполнен порог NDCG; звёздочка - сохранённый выбор. Отложенные периоды не влияют на настройку.')
        boundary = [dataset.upper() for dataset, result in self.results.items()
                    if result['selection']['reranker']['alpha'] == max(result['config']['rerank_alphas'])]
        if boundary:
            self.p('Для ' + ', '.join(boundary) + ' выбран α на верхней границе сетки. Это лучший допустимый '
                   'вариант среди проверенных, а не установленный глобальный оптимум коэффициента.', 'small')

    def hardware_supplement(self):
        path = self.evidence / 'hardware_replication.json'
        audit = self.read_input_json(path)
        self.page()
        self.h('Дополнение. Повтор третьего запуска KION на GPU')
        self.p('Основной эксперимент KION объединяет два GPU-запуска и один CPU-запуск RL. После фиксации '
               'этого маршрута появилась возможность завершить исходный третий GPU-запуск. Поэтому отдельно '
               'проверена чувствительность всех семи вариантов seed 33 к программной среде и оборудованию. '
               'Начальный SFT checkpoint, гиперпараметры, алгоритмический seed и исходники совпадают. '
               'Различаются Python/PyTorch и вычислительные операции CPU/CUDA; это не изолированный эффект устройства. '
               'Одинаковый seed также не гарантирует одинаковых траекторий семплирования на CPU и CUDA [9].')
        self.p('План этого описательного дополнения записан после начала основной оценки, до новых GPU-предсказаний. '
               'Обе версии оцениваются исходным кодом на CPU, на одних наблюдениях и с одним основным критиком. '
               'Выбор рецепта и основные результаты не заменяются. Новых тестов значимости, NI или подбора seed нет.')
        families = ['matched_control', 'raw_h5', 'anchored_h1', 'anchored_h5',
                    'appended_h5', 'conservative_h5', 'retention_h5']
        rows = []
        for family in families:
            row = [NAMES[family]]
            for split in ('test', 'confirmation'):
                for metric in ('ndcg@10', 'factual_completion_weighted_hit@10'):
                    value = audit['splits'][split]['families'][family]['metrics'][metric]['delta_gpu_minus_cpu_seed33']
                    row.append(f'{value:+.5f}')
            rows.append(row)
        self.table(['Вариант', 'ΔNDCG,<br/>тест', 'Δ r × Hit,<br/>тест',
                    'ΔNDCG,<br/>поздний', 'Δ r × Hit,<br/>поздний'], rows,
                   [.36, .16, .16, .16, .16],
                   'Таблица 15. GPU минус CPU для одного seed 33. Положительная разность означает большую метрику у GPU-версии. Это описательные разности двух checkpoints, без интервала по распределению обучения.')
        selected = self.results['kion']['selection']['selected_rl']
        for split, label in [('test', 'Основной тест'), ('confirmation', 'Подтверждение')]:
            metrics = audit['splits'][split]['families'][selected]['metrics']
            parts = []
            for metric, name in [('ndcg@10', 'NDCG'), ('factual_completion_weighted_hit@10', 'r × Hit')]:
                values = metrics[metric]
                parts.append(f"{name}: {number(values['primary_three_seed_mean'], 5)} → "
                             f"{number(values['sensitivity_replacement_mean'], 5)}")
            self.p(f'<b>{label}, фиксированный основной H5:</b> ' + '; '.join(parts) + '.', 'small')
        self.p('В этих последних числах показано, как изменилось бы среднее трёх запусков при замене только '
               'seed 33 на GPU-версию: изменение равно трети разности из таблицы. Это анализ чувствительности, '
               'а не четвёртый независимый SFT seed. Единственная пара сред не позволяет оценить надёжность '
               'обучения в целом. Сохранённые основные интервалы и NI относятся к исходной комбинации checkpoints.', 'small')

    def build(self):
        self.title()
        self.related()
        self.data()
        self.protocol()
        self.models()
        self.rl()
        self.reward_results()
        self.dataset_results('kion', 8)
        self.dataset_results('yambda', 9)
        self.tradeoff()
        self.behavior()
        self.longterm()
        self.ablations_and_feedback()
        self.conclusions()
        self.duration_supplement()
        self.hardware_supplement()
        self.appendix()
        self.output.parent.mkdir(parents=True, exist_ok=True)

        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Article', 8)
            if doc.page > 1:
                canvas.drawString(22 * mm, A4[1] - 15 * mm, 'RL для рекомендаций: KION и Yambda')
            canvas.drawCentredString(A4[0] / 2, 14 * mm, str(doc.page))
            canvas.restoreState()

        doc = SimpleDocTemplate(str(self.output), pagesize=A4, rightMargin=22 * mm, leftMargin=22 * mm,
                                topMargin=22 * mm, bottomMargin=22 * mm,
                                title='RL для рекомендаций: эксперимент на KION и Yambda', author='',
                                subject='Модельная награда, релевантность и многошаговая генерация')
        doc.build(self.story, onFirstPage=footer, onLaterPages=footer)
        return self.output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', default='results')
    parser.add_argument('--output', default='article.pdf')
    args = parser.parse_args()
    print(Article(args.evidence, args.output).build())
