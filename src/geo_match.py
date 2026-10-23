import os, sys
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # interactive session (e.g. Jupyter): run from the repository root
    _HERE = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, _HERE)
from config import *
import geopandas as gpd, pandas as pd, re, difflib
def norm(s):
    s=re.sub(r'\(.*?\)','',str(s).lower()).replace('&','and').replace('district','')
    return re.sub(r'[^a-z]','',s)
g=gpd.read_file(DISTRICTS_2011)
g['ks']=g.ST_NM.map(norm); g['kd']=g.DISTRICT.map(norm)
STATE_ALIAS={'arunanchalpradesh':'arunachalpradesh','andamanandnicobarisland':'andamanandnicobar','dadaraandnagarhavelli':'dadraandnagarhaveli','nctofdelhi':'delhi','jammuandkashmir':'jammuandkashmir','telangana':'andhrapradesh','orissa':'odisha','uttaranchal':'uttarakhand','pondicherry':'puducherry','nctofdelhi':'delhi',
             'dadranagarhaveli':'dadraandnagarhaveli','dadranagarhavelianddamananddiu':'dadraandnagarhaveli','ladakh':'jammuandkashmir','andamanandnicobarislands':'andamanandnicobar'}
DIST_ALIAS={  # renamed districts and post-2011 districts -> 2011 parent
 'khandwa':'eastnimar','khargone':'westnimar','narsinghpur':'narsimhapur','charkidadri':'bhiwani','fazilka':'firozpur','pathankot':'gurdaspur','malerkotla':'sangrur','charkhidadri':'bhiwani',
 'amethi':'sultanpur','hapur':'ghaziabad','sambhal':'moradabad','shamli':'muzaffarnagar','hathras':'mahamayanagar','agarmalwa':'shajapur','niwari':'tikamgarh',
 'palghar':'thane','beed':'bid','arvalli':'sabarkantha','aravalli':'sabarkantha','botad':'bhavnagar','chhotaudepur':'vadodara','dangs':'thedangs','dang':'thedangs',
 'devbhumidwarka':'jamnagar','girsomnath':'junagadh','mahisagar':'panchmahal','morbi':'rajkot','coochbehar':'kochbihar','purbabardhaman':'barddhaman',
 'paschimbardhaman':'barddhaman','bardhaman':'barddhaman','burdwan':'barddhaman','alipurduar':'jalpaiguri','kalimpong':'darjiling','jhargram':'paschimmedinipur',
 'anugola':'anugul','nayagada':'nayagarh','northgarohills':'eastgarohills','angul':'anugul','baragada':'baragarh','bargarh':'baragarh','debagada':'debagarh','deogarh':'debagarh','kataka':'cuttack','boudh':'baudh',
 'balod':'durg','balodabazarbhatapara':'raipur','balrampurramanujganj':'surguja','bemetara':'durg','gariyaband':'raipur','kondagaon':'bastar','mungeli':'bilaspur',
 'sukma':'dakshinbastardantewada','surajpur':'surguja','ysrkadapa':'ysr','sribhumi':'karimganj','bengalurusouth':'ramanagara','paurigarhwal':'garhwal',
 'eastsinghbum':'purbisinghbhum','siaha':'saiha','poonch':'punch','kasganj':'kanshiramnagar','prayagraj':'allahabad','ayodhya':'faizabad','gurugram':'gurgaon','nuh':'mewat','mewat':'mewat','belagavi':'belgaum','bengaluruurban':'bangalore','bengalururural':'bangalorerural',
 'mysuru':'mysore','kalaburagi':'gulbarga','vijayapura':'bijapur','ballari':'bellary','shivamogga':'shimoga','tumakuru':'tumkur','chikkamagaluru':'chikmagalur',
 'belgavi':'belgaum','prayagraj':'allahabad','ayodhya':'faizabad',
 'kasganj':'kanshiramnagar','ahilyanagar':'ahmadnagar','ahmednagar':'ahmadnagar','chhatrapatisambhajinagar':'aurangabad','dharashiv':'osmanabad',
 'narmadapuram':'hoshangabad',
 'sasnagar':'sahibzadaajitsinghnagar','mohali':'sahibzadaajitsinghnagar','nawanshahrsbsnagar':'shahidbhagatsinghnagar','nawanshahr':'shahidbhagatsinghnagar',
 'srimuktsarsahib':'muktsar','firozepur':'firozpur','ferozepur':'firozpur','kheri':'kheri','lakhimpurkheri':'kheri','kushinagar':'kushinagar','kushi':'kushinagar',
 'santkabeernagar':'santkabirnagar','mahrajganj':'mahrajganj','maharajganj':'mahrajganj','ambedkarnagar':'ambedkarnagar','jyotibaphulenagar':'jyotibaphulenagar',
 'amroha':'jyotibaphulenagar','panchmahals':'panchmahal','panchmahal':'panchmahal','dohad':'dohad','dahod':'dohad','banaskantha':'banaskantha','sabarkantha':'sabarkantha',
 'ahmedabad':'ahmadabad','kachchh':'kachchh','kutch':'kachchh','vadodara':'vadodara','thedangs':'dang','dangs':'dang','raigarh':'raigarh','raigad':'raigarh',
 'paschimmedinipur':'paschimmedinipur','purbamedinipur':'purbamedinipur','haora':'haora','howrah':'haora','hooghly':'hugli','hugli':'hugli',
 'cuttack':'cuttack','khordha':'khordha','khurda':'khordha','balasore':'baleshwar','baleswar':'baleshwar','kanyakumari':'kanniyakumari','tuticorin':'thoothukkudi',
 'thoothukudi':'thoothukkudi','tiruvallur':'thiruvallur','villupuram':'viluppuram','nilgiris':'thenilgiris','sivaganga':'sivaganga','kanchipuram':'kancheepuram',
 'spsrnellore':'srpottisriramulunellore','nellore':'srpottisriramulunellore','srpottisriramulunellore':'srpottisriramulunellore','ysr':'ysrkadapa','kadapa':'ysrkadapa','cuddapah':'ysrkadapa',
 'ananthapuramu':'anantapur','anantapuramu':'anantapur','anantapur':'anantapur','visakhapatanam':'visakhapatnam','rangareddy':'rangareddy','rangareddi':'rangareddy',
 'mahbubnagar':'mahbubnagar','mahabubnagar':'mahbubnagar','karimnagar':'karimnagar','sriganganagar':'ganganagar','ganganagar':'ganganagar','jhunjhunu':'jhunjhunun',
 'dholpur':'dhaulpur','chittorgarh':'chittaurgarh','leh':'leh','badgam':'badgam','budgam':'badgam','baramula':'baramula','baramulla':'baramula',
 'kamrup':'kamrup','kamrupmetro':'kamrupmetropolitan','sibsagar':'sivasagar','marigaon':'morigaon','dimahasao':'dimahasao','northcacharhills':'dimahasao',
 'kabirdham':'kabeerdham','kawardha':'kabeerdham','janjgir':'janjgirchampa','koriya':'koriya','korea':'koriya','dantewada':'dakshinbastardantewada','kanker':'uttarbastarkanker',
 'aurangabadbihar':'aurangabad','purbichamparan':'purbachamparan','eastchamparan':'purbachamparan','westchamparan':'pashchimchamparan',
 'pashchimichamparan':'pashchimchamparan','sheikhpura':'shekhpura','monghyr':'munger','saraikelakharsawan':'saraikelakharsawan','seraikela':'saraikelakharsawan',
 'eastsingbhum':'purbisinghbhum','purbisinghbhum':'purbisinghbhum','westsingbhum':'pashchimisinghbhum','pashchimisinghbhum':'pashchimisinghbhum','kodarma':'kodarma','koderma':'kodarma'}
def match_table(df, scol, dcol):
    """returns df with 2011 polygon censuscode (or None)"""
    look={s:dict(zip(gg.kd,gg.censuscode)) for s,gg in g.groupby('ks')}
    out=[]
    for s,d in zip(df[scol],df[dcol]):
        ks=STATE_ALIAS.get(norm(s),norm(s)); kd=norm(d); kd=DIST_ALIAS.get(kd,kd)
        c=look.get(ks,{})
        if kd in c: out.append((c[kd],'exact')); continue
        mm=difflib.get_close_matches(kd,list(c),n=1,cutoff=0.82)
        out.append((c[mm[0]],'fuzzy') if mm else (None,'none'))
    df=df.copy(); df['censuscode']=[o[0] for o in out]; df['geo_match']=[o[1] for o in out]; return df
