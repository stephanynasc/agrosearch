# Laboratório Prático 04 - Desafio Integrador
# Disciplina: Tópicos Avançados - Recuperação de Informação / PLN
# Professor: Me. Ricardo Roberto de Lima
#
# AgroSearch - Motor de Busca Inteligente
# Integra as 3 fases do desafio (Pré-processamento -> Índice Invertido -> TF-IDF)
# + Desafio Bônus (Similaridade de Cosseno).
#
# Integrantes do grupo:
# - <NOME 1>
# - <NOME 2>
# - <NOME 3>
#
# Restrição técnica atendida: nenhuma biblioteca de alto nível de NLP/ML
# (scikit-learn, TfidfVectorizer, nltk stemmers, etc.) é utilizada. Tokenização,
# normalização, stopwords, stemming, índice invertido, TF, IDF, TF-IDF e
# similaridade de cosseno são implementados do zero abaixo.

import streamlit as st
import pandas as pd
import re
import unicodedata
import math
from collections import defaultdict

st.set_page_config(page_title="AgroSearch", page_icon="🌱", layout="wide")

# ---------------------------------------------------------------------------
# BASE DE DOCUMENTOS
# 5 manuais técnicos reais fornecidos pelo professor (Doc1.pdf ... Doc5.pdf),
# simulando os manuais internos da AgroTech Solutions citados no estudo de caso.
# Cada documento guarda: título (exibição), resumo (prévia curta em tabelas)
# e texto (conteúdo integral, usado pelo pipeline de RI).
# ---------------------------------------------------------------------------
DOCUMENTOS = {
    "Doc 1": {
        "titulo": "Irrigação da Soja durante a Floração",
        "resumo": "Analisa se a soja realmente precisa de irrigação constante na floração, concluindo que a afirmação é apenas parcialmente correta.",
        "texto": """A soja (Glycine max) é uma das culturas agrícolas de maior importância econômica, sendo utilizada
principalmente na produção de alimentos, óleo e ração animal. Para que a planta alcance seu potencial
produtivo, é necessário que fatores como temperatura, fertilidade do solo, luminosidade e disponibilidade
de água estejam em condições adequadas. Entre as fases do ciclo da soja, o período reprodutivo é
especialmente importante, pois nele ocorre a formação das flores, vagens e grãos.

A água exerce papel fundamental no desenvolvimento da cultura. Ela participa da fotossíntese, do
transporte de nutrientes, da manutenção da estrutura celular e de diversas reações metabólicas. Por isso,
a ocorrência de déficit hídrico durante fases sensíveis pode provocar perdas de produtividade. Entretanto,
a afirmação de que a soja necessita de irrigação constante durante a floração deve ser analisada com
atenção, pois irrigar continuamente não significa necessariamente realizar um manejo adequado.

Durante a floração, a soja apresenta elevada demanda por água, uma vez que a planta está direcionando
energia para a formação das estruturas reprodutivas. Quando a disponibilidade hídrica é insuficiente, pode
ocorrer redução do crescimento e queda de flores e estruturas reprodutivas. Isso diminui a quantidade
potencial de vagens e pode afetar diretamente o rendimento da lavoura. A deficiência de água também pode
comprometer etapas posteriores do ciclo. Se o estresse hídrico permanecer durante a formação e o
enchimento dos grãos, a planta poderá produzir sementes menores e com menor peso. Dessa maneira, manter
uma disponibilidade adequada de água é essencial para que a soja consiga expressar seu potencial
produtivo.

A irrigação pode ser uma ferramenta importante em áreas sujeitas a períodos de estiagem ou chuvas
irregulares. Ao fornecer água quando a precipitação natural não é suficiente, o produtor consegue reduzir
os efeitos do déficit hídrico e aumentar a estabilidade da produção. O benefício, porém, depende de um
planejamento adequado, pois a quantidade de água necessária varia de acordo com o solo, o clima, a
cultivar e o estágio de desenvolvimento da planta.

Embora a soja necessite de água durante a floração, isso não significa que a irrigação deva ocorrer de
forma ininterrupta. A aplicação excessiva de água pode causar encharcamento do solo, reduzir a quantidade
de oxigênio disponível para as raízes e favorecer condições para o aparecimento de doenças. Além disso, o
uso desnecessário de água aumenta os custos de produção e pode causar desperdício de um recurso natural.
O manejo correto consiste em repor a água utilizada pela planta e perdida pelo solo, mantendo a umidade
em níveis adequados. Para isso, o produtor pode observar a umidade do solo, a ocorrência de chuvas e as
condições climáticas, além de utilizar métodos de estimativa da evapotranspiração e sensores de umidade,
que auxiliam na definição do momento e da quantidade de água a aplicar.

Um sistema de irrigação eficiente deve fornecer água de maneira uniforme e na quantidade necessária. O
manejo precisa considerar a capacidade do solo de armazenar água, a profundidade das raízes, a
temperatura, a umidade do ar e a demanda da cultura. Em períodos de chuva suficiente, a necessidade de
irrigação pode diminuir ou até desaparecer temporariamente. Já em períodos de estiagem, a irrigação
suplementar pode ser fundamental para evitar o estresse hídrico. A adoção de um manejo racional também
contribui para a sustentabilidade da produção agrícola, reduzindo o desperdício de água e energia e
diminuindo custos.

A afirmação apresentada é parcialmente correta. É correto dizer que a soja possui elevada necessidade de
água durante a floração e que a falta de água nesse período pode reduzir a produtividade. Contudo, não é
correto afirmar que a cultura necessita obrigatoriamente de irrigação constante. A necessidade de
irrigação depende da disponibilidade de água no solo e das condições ambientais. Portanto, o objetivo do
produtor não deve ser manter o solo permanentemente molhado, mas evitar que a cultura passe por estresse
hídrico. O manejo deve buscar equilíbrio entre disponibilidade e consumo de água.

A água é um dos fatores fundamentais para a produtividade da soja, especialmente durante o período de
floração e nas demais fases reprodutivas. A deficiência hídrica pode provocar queda de flores, redução do
número de vagens e prejuízos ao enchimento dos grãos. Por isso, garantir adequada disponibilidade de água
é essencial para o desenvolvimento da cultura. Entretanto, a irrigação constante não é necessariamente
recomendada. O mais adequado é realizar um manejo baseado na necessidade real da planta, nas condições
do solo e no clima. Dessa forma, a irrigação planejada pode contribuir para maior produtividade, maior
eficiência no uso da água e uma produção agrícola mais sustentável.""",
    },
    "Doc 2": {
        "titulo": "Controle Biológico de Lagartas na Soja com Vespas Trichogramma",
        "resumo": "Descreve o uso de vespas parasitoides Trichogramma pretiosum no controle biológico de lagartas desfolhadoras da soja.",
        "texto": """A cultura da soja (Glycine max) representa um dos pilares mais sólidos do agronegócio global. Contudo,
a estabilidade e a produtividade dessa lavoura são constantemente ameaçadas por complexos de pragas
desfolhadoras e pragas de vagens. Historicamente, o uso contínuo de químicos resultou na seleção de
insetos resistentes e na redução de inimigos naturais nativos.

Nesse cenário, o Controle Biológico surge como uma estratégia indispensável no Manejo Integrado de Pragas
(MIP). Dentre as ferramentas de biocontrole, a utilização de microvespas parasitoides do gênero
Trichogramma, especialmente Trichogramma pretiosum, consolidou-se como uma solução altamente eficiente,
sustentável e acessível para conter as infestações de lepidópteros na soja. O uso de vespinhas
parasitoides reduz a dependência de inseticidas neurotóxicos, preserva polinizadores e ajuda a manter o
equilíbrio ecológico dos solos e da fauna auxiliar.

O combate preventivo aos ovos impede a eclosão e o consequente dano foliar ou de vagens. As principais
pragas controladas por esse método incluem a lagarta-da-soja (Anticarsia gemmatalis), desfolhadora
tradicional responsável por severa redução da área fotossintética; a lagarta-falsa-medideira (Chrysodeixis
includens), que raspa as folhas deixando o aspecto rendilhado e apresenta alta tolerância a químicos
comuns; o complexo Helicoverpa (H. armigera e H. zea), pragas vorazes que atacam estruturas vegetativas e
reprodutivas como flores e vagens; e o complexo Spodoptera (S. frugiperda, S. eridania, S. cosmioides),
desfolhadoras e cortadoras de plantas jovens.

A microvespa Trichogramma pretiosum é um endoparasitoide de ovos de apenas 0,5 mm. Seu comportamento
especializado ataca a praga no início do ciclo: a fêmea adulta localiza ovos recém-depositados pelas
mariposas nas folhas, insere seu ovipositor na casca do ovo e deposita seu próprio ovo em seu interior;
substâncias injetadas interrompem o desenvolvimento do embrião da lagarta; a larva da vespa consome o
conteúdo interno e o ovo parasitado torna-se escuro; a nova vespa adulta emerge e reinicia o ciclo no
campo. A taxa de parasitismo no campo é calculada pela relação entre ovos parasitados e a densidade total
de ovos. Em liberações bem executadas, a eficiência supera 80%, neutralizando a praga antes que ocorram
danos foliares.

A liberação deve ocorrer quando as mariposas adultas começam a ser detectadas em armadilhas de feromônio,
no início das posturas. A dosagem média recomendada é de 100.000 a 200.000 vespas por hectare, divididas
em 2 a 3 aplicações, com intervalos de 7 a 10 dias após o início da postura de mariposas. A liberação pode
ser feita por drones, cápsulas biodegradáveis ou cartelas, preferencialmente em temperaturas entre 20°C e
30°C, evitando calor extremo e ventos fortes. O emprego de drones na agricultura de precisão permite
distribuir pupas prestes a emergir com grande velocidade e uniformidade, viabilizando o biocontrole em
grandes extensões com baixo custo operacional.

O sucesso do biocontrole exige o respeito à seletividade dos defensivos agrícolas. Inseticidas de amplo
espectro, como piretroides e organofosforados, podem eliminar as vespas. Quando houver necessidade de
intervir quimicamente contra outras pragas, como percevejos, deve-se optar por produtos seletivos ou
reguladores de crescimento de insetos.

Entre os benefícios econômicos e ambientais do método estão a destruição do ovo antes do nascimento da
lagarta, a redução de resíduos que facilita a produção de grãos limpos para exportação, o retardo do
surgimento de populações de insetos resistentes a químicos e a viabilidade econômica, com custo
competitivo via aplicação com drones comparado a pulverizações convencionais.""",
    },
    "Doc 3": {
        "titulo": "Adubação Verde com Leguminosas para o Milho",
        "resumo": "Explica como leguminosas usadas em adubação verde fixam nitrogênio biologicamente para nutrir a cultura do milho.",
        "texto": """A cultura do milho (Zea mays) possui uma das maiores demandas nutricionais de nitrogênio entre as
grandes culturas anuais. O nitrogênio é o elemento determinante para o desenvolvimento vegetativo,
expansão foliar, síntese de proteínas e formação dos grãos. Historicamente, o suprimento desse nutriente
é dependente de fertilizantes nitrogenados sintéticos de alto custo, como ureia e nitrato de amônio,
sujeitos a perdas por lixiviação, volatilização e desnitrificação.

Nesse cenário, a adubação verde com leguminosas surge como uma prática agroecológica e biologicamente
sustentável. Essa técnica consiste na inserção de plantas da família Fabaceae em rotação ou sucessão de
culturas, visando à Fixação Biológica de Nitrogênio (FBN) e à incorporação de matéria orgânica de rápida
decomposição no solo. Leguminosas adubadoras podem aportar ao solo de 100 a mais de 250 kg de nitrogênio
por hectare por via biológica, reduzindo significativamente a dependência de fertilizantes químicos na
cultura subsequente do milho.

A escolha da espécie de leguminosa depende da região fitogeográfica, época do ano e disponibilidade
hídrica. As crotalárias (Crotalaria juncea, C. spectabilis, C. ochroleuca) apresentam excelente produção
de biomassa e alta taxa de fixação biológica de nitrogênio, além de propriedades nematicidas reconhecidas.
O guandu (Cajanus cajan) é uma leguminosa de porte arbustivo com sistema radicular profundo e pivotante,
capaz de romper camadas compactadas e reciclar nutrientes de horizontes profundos. As mucunas (Mucuna
pruriens, M. aterrima) proporcionam excelente cobertura vegetal e supressão de plantas daninhas, além de
expressiva adição de matéria orgânica rica em nitrogênio. A vica-sativa (Vicia sativa) e a ervilhaca são
bastante empregadas na região Sul durante o outono e inverno, precedendo o plantio de milho na primavera.

A associação simbiótica entre as raízes das leguminosas e bactérias diazotróficas dos gêneros Rhizobium e
Bradyrhizobium permite converter o nitrogênio atmosférico, indisponível para as plantas, em formas
assimiláveis como a amônia. No manejo da adubação verde, as bactérias nodulam nas raízes da leguminosa e
fixam o nitrogênio atmosférico; o nitrogênio é incorporado na biomassa da planta; após a roçada ou o
acamamento com rolo-faca, a palhada fica sobre o solo; com baixa relação carbono/nitrogênio, a
decomposição microbiana é rápida; e ocorre a mineralização, liberando formas de nitrogênio exatamente no
período de maior exigência do milho. A taxa de mineralização em leguminosas costuma variar entre 60% e
80% no primeiro ciclo agrícola.

Para sincronizar a liberação de nitrogênio com a fase de maior absorção do milho, o manejo da leguminosa
deve ser planejado com precisão, preferencialmente no pico de florescimento, momento em que a planta
atinge o máximo acúmulo de biomassa e teor de nitrogênio, evitando a lignificação excessiva dos tecidos.
A crotalaria juncea, por exemplo, produz de 6 a 10 toneladas de massa seca por hectare e aporta de 150 a
250 kg de nitrogênio por hectare quando manejada no pleno florescimento.

Embora a melhoria dos teores de nitrogênio seja o principal foco, a adubação verde proporciona benefícios
sistêmicos ao solo: melhoria estrutural, com descompactação biológica promovida pelas raízes profundas e
aumento da infiltração de água; aumento da matéria orgânica, incrementando a capacidade de troca catiônica
e a retenção hídrica; proteção contra erosão, pois a palhada protege o solo do impacto direto das gotas de
chuva; e estímulo à atividade biológica, favorecendo fungos micorrízicos e bactérias benéficas da
rizosfera.

A substituição parcial ou total da adubação nitrogenada mineral por nitrogênio biológico proveniente das
leguminosas proporciona uma economia expressiva na aquisição de insumos sintéticos. A adubação verde com
leguminosas consolida-se como um pilar fundamental para uma agricultura de alto rendimento, resiliência
climática e baixo impacto ambiental na milhicultura.""",
    },
    "Doc 4": {
        "titulo": "Manejo de Lagartas Desfolhadoras na Soja e no Algodão",
        "resumo": "Aborda as principais espécies de lagartas desfolhadoras, seus danos e as estratégias de manejo integrado na soja e no algodão.",
        "texto": """As culturas da soja (Glycine max) e do algodão (Gossypium hirsutum) representam dois dos pilares mais
estratégicos do agronegócio brasileiro. Contudo, a sustentabilidade econômica dessas lavouras é
constantemente ameaçada pelo ataque de pragas complexas. Dentre elas, as lagartas desfolhadoras se
destacam pelo elevado potencial de destruição da área foliar, comprometendo a taxa fotossintética e
reduzindo drasticamente a produtividade final dos grãos e das plumas. Devido ao sistema de cultivo em
sucessão, muitas dessas espécies de lepidópteros encontram ponte verde e abrigo contínuo ao longo do ano,
aumentando a densidade populacional e a pressão de seleção sobre os métodos tradicionais de controle.
Infestações severas não controladas de lagartas desfolhadoras podem causar reduções superiores a 40% no
rendimento da soja e perda total da qualidade das maçãs e plumas no algodoeiro.

Tanto na soja quanto no algodão, o complexo de lagartas envolve espécies com hábitos alimentares e níveis
de suscetibilidade distintos aos defensivos. A lagarta-falsa-medideira (Chrysodeixis includens) ataca
preferencialmente o terço médio e inferior das plantas, raspando o parênquima foliar e deixando as
nervuras intactas, com alta capacidade de surtos e tolerância a diversos inseticidas. A lagarta-da-soja
(Anticarsia gemmatalis) consome todo o limbo foliar e, em altas populações, pode causar desfolha total em
poucos dias. O complexo Spodoptera (S. frugiperda, S. eridania, S. cosmioides) é extremamente agressivo:
na soja atua como desfolhadora e cortadora, e no algodão ataca folhas, botões florais e roem as maçãs. O
complexo Helicoverpa (H. armigera e H. zea) é polífago e voraz, alimentando-se das folhas jovens e
brotações antes de migrar rapidamente para as estruturas reprodutivas, como vagens na soja e
maçãs/capulhos no algodão.

A perda da área foliar reduz a interceptação de radiação solar e interfere diretamente na translocação de
fotoassimilados para o enchimento de grãos na soja e a formação de fibras no algodão. As culturas têm
tolerâncias diferentes à desfolha: a soja apresenta boa capacidade de compensação vegetativa, tolerando
até 30% de desfolha na fase vegetativa e 15% na fase reprodutiva; já o algodão, de arquitetura mais
complexa e ciclo longo, tolera menos desfolha, entre 10% e 15%, pois qualquer perda foliar severa induz o
abortamento de botões florais e estruturas reprodutivas. O cálculo do índice de área foliar remanescente e
do nível de dano econômico orienta a tomada de decisão para o controle.

Na soja, a fase mais crítica vai de R1 a R5 (floração ao enchimento), com sintomas de folhas rendilhadas
ou devoradas, e o nível de ação recomendado é de 20 lagartas grandes por metro ou 15% de desfolha no
reprodutivo. No algodão, a fase mais crítica vai do botão floral ao capulho (B1 a C1), com desfolha
superior e perfuração de botões e maçãs, e o nível de ação recomendado é de 10% de plantas com desfolha ou
presença de 1 lagarta por metro de linha.

Para conter os prejuízos causados pelas desfolhadoras de forma sustentável, é indispensável adotar o
Manejo Integrado de Pragas: monitoramento constante, com pano de batida na soja e vistoria de plantas no
algodão, associados a armadilhas de feromônio para mariposas; tecnologia Bt, com cultivares expressando
proteínas de Bacillus thuringiensis que conferem resistência às principais lagartas; manejo de resistência,
com plantio obrigatório de áreas de refúgio estruturado para preservar a suscetibilidade das pragas à
tecnologia; controle biológico, com parasitoides de ovos como Trichogramma spp., predadores como
percevejos Podisus e biopesticidas à base de Bacillus thuringiensis ou Baculovírus; e controle químico
seletivo, com inseticidas fisiológicos que preservam os inimigos naturais.

O combate às lagartas desfolhadoras na soja e no algodão exige planejamento integrativo e monitoramento
constante. A dependência exclusiva de uma única ferramenta aumenta o risco de falhas de controle e rápida
seleção de populações resistentes. A combinação harmônica entre biotecnologia, biocontrole e químicos
seletivos é o único caminho para assegurar altas produtividades e rentabilidade ao produtor rural.""",
    },
    "Doc 5": {
        "titulo": "Irrigação por Gotejamento e Cultivo Orgânico",
        "resumo": "Apresenta a eficiência hídrica da irrigação por gotejamento e sua sinergia com os princípios do cultivo orgânico.",
        "texto": """A gestão sustentável dos recursos hídricos tornou-se um dos maiores desafios da agricultura moderna.
No modelo de produção orgânica, cujo foco central reside na preservação dos ecossistemas, na saúde do solo
e no uso racional dos insumos, a escolha do sistema de irrigação desempenha um papel determinante. A
irrigação por gotejamento destaca-se como o método mais eficiente e compatível com os princípios da
agroecologia. Por meio da aplicação localizada de água diretamente na zona radicular das plantas, o
sistema reduz o desperdício, preserva a estrutura biológica do solo e minimiza as condições favoráveis ao
surgimento de doenças fitopatológicas. Enquanto sistemas convencionais por aspersão apresentam eficiência
entre 60% e 75%, a irrigação por gotejamento atinge uma eficiência de aplicação superior a 90% e 95%,
reduzindo drasticamente o consumo de água.

O gotejamento é um sistema de irrigação localizada de alta frequência e baixa pressão. A água é
distribuída por tubos gotejadores equipados com emissores regulados que liberam gotas em vazões
controladas, geralmente de 1,0 a 4,0 litros por hora. Entre os fatores da economia hídrica estão a redução
da evaporação direta, já que a água é aplicada em pontos específicos cobrindo apenas a zona radicular; a
eliminação da perda por vento e deriva, diferente da aspersão, que não forma névoa suscetível à ação do
vento; e o controle de percolação profunda, já que a água é fornecida na lâmina exata exigida pela
cultura, evitando que percole além da profundidade das raízes. A Eficiência de Uso da Água resulta em um
rendimento por metro cúbico substancialmente superior quando comparado aos métodos tradicionais de
irrigação por superfície ou aspersão.

O uso da irrigação por gotejamento no cultivo orgânico vai além da simples economia de água, sendo um
componente chave no manejo fitossanitário e nutricional. No manejo de doenças fúngicas e bacterianas, como
as folhas das plantas permanecem secas, ao contrário da aspersão, cria-se um microclima desfavorável à
germinação de esporos de fungos foliares como míldio, oídio e requeima, reduzindo a necessidade de
tratamentos cúpricos ou biológicos. Na supressão de plantas espontâneas, a aplicação de água restringe-se
ao bulbo úmido junto à cultura principal, deixando as entrelinhas secas e inibindo a germinação de ervas
invasoras sem a necessidade de herbicidas sintéticos. A biofertirrigação possibilita a aplicação de
insumos orgânicos líquidos solúveis, como biofertilizantes fermentados, extratos compostos e inoculantes
microbiológicos, diretamente na raiz das plantas. Além disso, o sistema preserva a estrutura do solo,
evitando o selamento superficial e a erosão causados pela gota da chuva ou de aspersores, mantendo a
porosidade e a aeração ideais para a microbiota benéfica.

Comparando as tecnologias de irrigação aplicadas à horticultura e fruticultura orgânica: o gotejamento
localizado atinge de 90% a 95% de eficiência hídrica, com molhamento foliar nulo, desenvolvimento de
daninhas muito baixo e compatibilidade orgânica excelente; a microaspersão atinge de 80% a 85%, com
molhamento foliar baixo a médio e compatibilidade boa; a aspersão convencional atinge de 65% a 75%, com
molhamento foliar alto, desenvolvimento de daninhas elevado e compatibilidade apenas razoável, devido ao
risco de doenças; e a irrigação por sulcos ou superfície atinge de 40% a 60%, com desenvolvimento de
daninhas muito elevado e baixa compatibilidade orgânica, devido ao risco de erosão.

Para garantir a longevidade e o pleno funcionamento das linhas gotejadoras no cultivo orgânico, recomenda-
se filtragem eficiente, com sistemas de filtros de disco ou de areia para evitar o entupimento dos
emissores; monitoramento da qualidade da água, observando teores de ferro, cálcio e bicarbonatos que
possam precipitar e obstruir os gotejadores; uso de biofertilizantes filtrados, passando toda solução
biofertilizante por peneiramento e filtragem fina prévia; e uso de cobertura morta (mulching), combinando
o gotejamento com a palhada para reduzir a evaporação a praticamente zero.

A irrigação por gotejamento consolida-se como a tecnologia indispensável para a consolidação de uma
agricultura orgânica moderna, produtiva e ecologicamente responsável. Ao aliar a máxima eficiência no uso
da água com a redução expressiva da incidência de pragas e doenças, o sistema protege os ecossistemas
hídricos e eleva a rentabilidade e sustentabilidade do produtor rural.""",
    },
}

STOPWORDS_PT = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "em", "para", "com",
    "um", "uma", "uns", "umas", "que", "se", "e", "ou", "por", "no", "na",
    "nos", "nas", "ao", "aos", "sua", "seu", "suas", "seus", "mais", "muito",
    "ja", "tambem", "como", "mas", "este", "esta", "isso", "e",
}

# ---------------------------------------------------------------------------
# FASE 1: PIPELINE DE PRÉ-PROCESSAMENTO (implementado do zero)
# ---------------------------------------------------------------------------


def tokenizar(texto):
    """Divide o texto em tokens (palavras), ignorando pontuação."""
    return re.findall(r"\w+", texto, flags=re.UNICODE)


def normalizar(token):
    """Lowercase + remoção de acentuação (Unicode NFD -> ASCII)."""
    token = token.lower()
    token = unicodedata.normalize("NFD", token).encode("ascii", "ignore").decode("utf-8")
    return token


# Sufixos ordenados dos mais específicos para os mais genéricos: o primeiro
# que casar (mantendo um radical de tamanho mínimo) é aplicado.
SUFIXOS_STEM = [
    "izacoes", "acoes", "amento", "imento", "idade", "mente",
    "acao", "icao",
    "adoras", "adores", "istas", "ancia", "encia",
    "ico", "ica", "icos", "icas", "oso", "osa", "osas", "osos",
    "ivo", "iva", "ivos", "ivas", "ado", "ada",
    "ar", "er", "ir",
    "as", "es", "os", "ao",
    "a", "o", "e", "s",
]


def stem_rudimentar(token, tamanho_minimo=3):
    """Stemmer heurístico simples: corta o primeiro sufixo da lista que
    casar, desde que o radical resultante mantenha um tamanho mínimo."""
    for sufixo in SUFIXOS_STEM:
        if token.endswith(sufixo) and len(token) - len(sufixo) >= tamanho_minimo:
            return token[: -len(sufixo)]
    return token


def preprocessar(texto, aplicar_stopwords=True, aplicar_stemming=True):
    """Executa o pipeline completo e retorna cada etapa (para fins didáticos)."""
    etapa_tokens = tokenizar(texto)
    etapa_normalizado = [normalizar(t) for t in etapa_tokens]

    if aplicar_stopwords:
        etapa_sem_stopwords = [t for t in etapa_normalizado if t not in STOPWORDS_PT]
    else:
        etapa_sem_stopwords = etapa_normalizado[:]

    if aplicar_stemming:
        etapa_final = [stem_rudimentar(t) for t in etapa_sem_stopwords]
    else:
        etapa_final = etapa_sem_stopwords[:]

    return {
        "tokens": etapa_tokens,
        "normalizado": etapa_normalizado,
        "sem_stopwords": etapa_sem_stopwords,
        "final": etapa_final,
    }


# ---------------------------------------------------------------------------
# FASE 2: ÍNDICE INVERTIDO (em memória, do zero)
# ---------------------------------------------------------------------------


def construir_indice_invertido(documentos, aplicar_stopwords, aplicar_stemming):
    indice = defaultdict(set)
    tokens_por_doc = {}
    for doc_id, info in documentos.items():
        tokens_finais = preprocessar(info["texto"], aplicar_stopwords, aplicar_stemming)["final"]
        tokens_por_doc[doc_id] = tokens_finais
        for termo in tokens_finais:
            indice[termo].add(doc_id)
    indice_ordenado = {termo: sorted(docs) for termo, docs in sorted(indice.items())}
    return indice_ordenado, tokens_por_doc


# ---------------------------------------------------------------------------
# FASE 3: TF-IDF e RANQUEAMENTO (do zero)
# ---------------------------------------------------------------------------


def calcular_tf(termo, tokens_doc):
    if not tokens_doc:
        return 0.0
    return tokens_doc.count(termo) / len(tokens_doc)


def calcular_idf(termo, tokens_por_doc):
    N = len(tokens_por_doc)
    df = sum(1 for tokens in tokens_por_doc.values() if termo in tokens)
    if df == 0:
        return 0.0
    return math.log(N / df)


def vocabulario(tokens_por_doc):
    termos = set()
    for tokens in tokens_por_doc.values():
        termos.update(tokens)
    return sorted(termos)


def vetor_tfidf_documento(doc_id, tokens_por_doc, vocab, idf_cache):
    tokens_doc = tokens_por_doc[doc_id]
    return [calcular_tf(t, tokens_doc) * idf_cache[t] for t in vocab]


def vetor_tfidf_query(tokens_query, vocab, idf_cache):
    return [calcular_tf(t, tokens_query) * idf_cache.get(t, 0.0) for t in vocab]


def similaridade_cosseno(v1, v2):
    """Similaridade de cosseno entre dois vetores (implementada do zero,
    sem numpy/sklearn)."""
    produto_escalar = sum(a * b for a, b in zip(v1, v2))
    norma1 = math.sqrt(sum(a * a for a in v1))
    norma2 = math.sqrt(sum(b * b for b in v2))
    if norma1 == 0 or norma2 == 0:
        return 0.0
    return produto_escalar / (norma1 * norma2)


# ---------------------------------------------------------------------------
# INTERFACE STREAMLIT
# ---------------------------------------------------------------------------

st.title("🌱 AgroSearch — Motor de Busca Inteligente")
st.caption(
    "AgroTech Solutions · Protótipo de motor de busca textual para manuais "
    "técnicos de campo (agricultura sustentável, pragas e irrigação)"
)

with st.sidebar:
    st.header("⚙️ Configurações do Pipeline")
    aplicar_stopwords = st.checkbox("Remover Stopwords", value=True)
    aplicar_stemming = st.checkbox("Aplicar Stemming", value=True)
    st.divider()
    usar_cosseno = st.checkbox("🎁 Usar Similaridade de Cosseno (Bônus)", value=True)
    st.divider()
    st.markdown("**Base de documentos**")
    st.caption("5 manuais técnicos da AgroTech (texto completo na aba 📄 Documentos).")
    for doc_id, info in DOCUMENTOS.items():
        st.caption(f"**{doc_id} — {info['titulo']}**")

# O índice e o vocabulário são recalculados a cada interação, refletindo
# dinamicamente os toggles de stopwords/stemming da barra lateral.
indice_invertido, tokens_por_doc = construir_indice_invertido(
    DOCUMENTOS, aplicar_stopwords, aplicar_stemming
)
vocab = vocabulario(tokens_por_doc)
idf_cache = {termo: calcular_idf(termo, tokens_por_doc) for termo in vocab}

tab0, tab1, tab2, tab3 = st.tabs(
    ["📄 Documentos", "1️⃣ Pré-processamento", "2️⃣ Índice Invertido", "3️⃣ Busca & Ranking"]
)

# --- BASE DE DOCUMENTOS -----------------------------------------------------
with tab0:
    st.subheader("Base de Documentos da AgroTech Solutions")
    st.write("Manuais técnicos completos usados como coleção (corpus) para a busca.")
    for doc_id, info in DOCUMENTOS.items():
        n_palavras = len(info["texto"].split())
        with st.expander(f"{doc_id} — {info['titulo']}  ·  ~{n_palavras} palavras"):
            st.caption(info["resumo"])
            st.write(info["texto"])

# --- FASE 1 ---------------------------------------------------------------
with tab1:
    st.subheader("Pipeline de Pré-processamento de Texto")
    st.write(
        "Veja o vocabulário mudar dinamicamente conforme você liga/desliga "
        "**Stopwords** e **Stemming** na barra lateral."
    )
    doc_escolhido = st.selectbox(
        "Escolha um documento para inspecionar:",
        list(DOCUMENTOS.keys()),
        format_func=lambda d: f"{d} — {DOCUMENTOS[d]['titulo']}",
    )
    resultado = preprocessar(DOCUMENTOS[doc_escolhido]["texto"], aplicar_stopwords, aplicar_stemming)

    c1, c2 = st.columns(2)
    with c1:
        with st.expander("1. Tokenização", expanded=True):
            st.write(resultado["tokens"])
        with st.expander("2. Normalização (minúsculas + sem acentos)", expanded=True):
            st.write(resultado["normalizado"])
    with c2:
        with st.expander(f"3. Remoção de Stopwords {'✅' if aplicar_stopwords else '(desligado)'}", expanded=True):
            st.write(resultado["sem_stopwords"])
        with st.expander(f"4. Stemming {'✅' if aplicar_stemming else '(desligado)'}", expanded=True):
            st.write(resultado["final"])

    st.divider()
    st.markdown("**Vocabulário final de todos os documentos (configuração atual):**")
    st.write(vocab)
    st.caption(f"Tamanho do vocabulário: {len(vocab)} termos únicos")

# --- FASE 2 -----------------------------------------------------------------
with tab2:
    st.subheader("Índice Invertido (Termo → Documentos)")
    st.write(
        "Construído em memória a partir dos tokens já pré-processados "
        "(reflete a configuração de Stopwords/Stemming escolhida ao lado)."
    )

    formato = st.radio("Formato de exibição:", ["Tabela", "JSON"], horizontal=True)
    if formato == "JSON":
        st.json(indice_invertido)
    else:
        linhas = [
            {"Termo": termo, "Documentos": ", ".join(docs), "DF (nº de docs)": len(docs)}
            for termo, docs in indice_invertido.items()
        ]
        st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

    st.caption(f"{len(indice_invertido)} termos únicos indexados em {len(DOCUMENTOS)} documentos.")

# --- FASE 3 -----------------------------------------------------------------
with tab3:
    st.subheader("Busca e Ranqueamento por TF-IDF")
    query = st.text_input("Digite sua consulta:", "irrigação soja")

    if st.button("🔍 Buscar", type="primary"):
        tokens_query = preprocessar(query, aplicar_stopwords, aplicar_stemming)["final"]
        termos_query_unicos = sorted(set(tokens_query))

        if not termos_query_unicos:
            st.warning("A consulta ficou vazia após o pré-processamento (talvez só continha stopwords).")
        else:
            st.markdown(f"**Termos da consulta após pré-processamento:** `{termos_query_unicos}`")

            termos_fora_vocab = [t for t in termos_query_unicos if t not in vocab]
            if termos_fora_vocab:
                st.caption(f"⚠️ Termos não encontrados em nenhum documento: {termos_fora_vocab}")

            # Detalhamento TF / IDF / TF-IDF por termo x documento
            detalhamento = []
            score_acumulado = {doc_id: 0.0 for doc_id in DOCUMENTOS}
            for termo in termos_query_unicos:
                idf = idf_cache.get(termo, 0.0)
                for doc_id in DOCUMENTOS:
                    tf = calcular_tf(termo, tokens_por_doc[doc_id])
                    tfidf = tf * idf
                    score_acumulado[doc_id] += tfidf
                    detalhamento.append(
                        {
                            "Termo": termo,
                            "Documento": doc_id,
                            "TF": round(tf, 4),
                            "IDF": round(idf, 4),
                            "TF-IDF": round(tfidf, 4),
                        }
                    )

            with st.expander("Ver detalhamento TF / IDF / TF-IDF por termo"):
                st.dataframe(pd.DataFrame(detalhamento), use_container_width=True, hide_index=True)

            ranking = (
                pd.DataFrame(
                    [
                        {
                            "Documento": doc_id,
                            "Título": DOCUMENTOS[doc_id]["titulo"],
                            "TF-IDF Acumulado": round(score, 4),
                        }
                        for doc_id, score in score_acumulado.items()
                    ]
                )
                .sort_values("TF-IDF Acumulado", ascending=False)
                .reset_index(drop=True)
            )
            ranking.insert(0, "Posição", ranking.index + 1)
            if ranking.loc[0, "TF-IDF Acumulado"] > 0:
                ranking.loc[0, "Documento"] = "🏆 " + ranking.loc[0, "Documento"]

            st.markdown("### Ranking Final (TF-IDF)")
            st.dataframe(ranking, use_container_width=True, hide_index=True)

            vencedor = ranking.iloc[0]
            if vencedor["TF-IDF Acumulado"] > 0:
                st.success(f"📌 Documento mais relevante: **{vencedor['Documento']}** — \"{vencedor['Título']}\"")
            else:
                st.warning("Nenhum documento contém os termos buscados.")

            # --- BÔNUS: Similaridade de Cosseno --------------------------
            if usar_cosseno:
                st.divider()
                st.markdown("### 🎁 Ranking Bônus: Similaridade de Cosseno")
                st.caption(
                    "Compara o vetor TF-IDF da consulta com o vetor TF-IDF de cada "
                    "documento em todo o vocabulário — lida melhor com consultas "
                    "de múltiplas palavras do que a simples soma de TF-IDF."
                )

                vetor_query = vetor_tfidf_query(tokens_query, vocab, idf_cache)
                linhas_cos = []
                for doc_id in DOCUMENTOS:
                    v_doc = vetor_tfidf_documento(doc_id, tokens_por_doc, vocab, idf_cache)
                    sim = similaridade_cosseno(vetor_query, v_doc)
                    linhas_cos.append(
                        {
                            "Documento": doc_id,
                            "Título": DOCUMENTOS[doc_id]["titulo"],
                            "Similaridade de Cosseno": round(sim, 4),
                        }
                    )

                ranking_cos = (
                    pd.DataFrame(linhas_cos)
                    .sort_values("Similaridade de Cosseno", ascending=False)
                    .reset_index(drop=True)
                )
                ranking_cos.insert(0, "Posição", ranking_cos.index + 1)
                if ranking_cos.loc[0, "Similaridade de Cosseno"] > 0:
                    ranking_cos.loc[0, "Documento"] = "🏆 " + ranking_cos.loc[0, "Documento"]

                st.dataframe(ranking_cos, use_container_width=True, hide_index=True)

                vencedor_cos = ranking_cos.iloc[0]
                if vencedor_cos["Similaridade de Cosseno"] > 0:
                    st.info(f"🧭 Pelo cosseno, o documento mais próximo da consulta é **{vencedor_cos['Documento']}**.")

st.divider()
st.caption(
    "Desenvolvido para o Laboratório Prático 04 · Tópicos Avançados - RI/PLN · "
    "Prof. Me. Ricardo Roberto de Lima"
)
