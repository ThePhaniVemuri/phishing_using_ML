# app.py
import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re
import socket
import math
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Constants for phishing detection
SUSPICIOUS_TLDS = {'xyz', 'tk', 'ml', 'ga', 'cf', 'gq', 'pw', 'cc', 'gb', 'top', 'work', 'party', 'date', 'trade', 'science', 'win'}

BRAND_NAMES = {
    'paypal', 'google', 'facebook', 'apple', 'microsoft', 'twitter', 'instagram',
    'amazon', 'ebay', 'netflix', 'linkedin', 'bankofamerica', 'chase', 'wellsfargo',
    'citibank', 'dropbox', 'yahoo', 'outlook', 'office365', 'whatsapp'
}

SENSITIVE_WORDS = {
    'login', 'secure', 'security', 'update', 'verify', 'verification', 'account',
    'signin', 'register', 'password', 'credential', 'confirm', 'alert', 'authenticate'
}

st.set_page_config(page_title="Phishing URL Checker", layout="wide")

st.title("🔎 Phishing URL Checker (ML)")
st.write("Enter a URL and the app will analyze features (from URL + page) and predict whether it's phishing or legitimate.")

DATA_CSV = "phishing_dataset.csv"   # your dataset filename
MODEL_FILE = "rf_phishing_model.pkl"

# ---------------------------
# Utility feature functions
# ---------------------------
def is_ip_address(hostname):
    try:
        # IPv4 or IPv6 detection (simple)
        socket.inet_aton(hostname)
        return 1
    except Exception:
        return 0

def count_dots(netloc):
    return netloc.count('.')

def subdomain_level(hostname, domain):
    # number of parts before registered domain; cheap heuristic:
    parts = hostname.split('.')
    # try to remove known TLDs? simple: return len(parts)-2
    return max(0, len(parts) - 2)

def path_level(path):
    if not path or path == '/':
        return 0
    return path.count('/')

def num_chars(s):
    return sum(c.isdigit() for c in s)

def has_at_symbol(url):
    return 1 if '@' in url else 0

def has_tilde(url):
    return 1 if '~' in url else 0

def count_char(s, ch):
    return s.count(ch)

def url_length(url):
    return len(url)

def hostname_length(host):
    return len(host)

def path_length(path):
    return len(path)

def query_length(query):
    return len(query)

def double_slash_in_path(path):
    return 1 if '//' in path and not path.startswith('//') else 0

def detect_random_string(segment):
    """Enhanced random string detection"""
    if not segment:
        return 0
    
    # Check length
    if len(segment) >= 10:
        # Check character diversity
        char_ratio = len(set(segment)) / len(segment)
        if char_ratio > 0.7:
            return 1
            
    # Check digit ratio
    digits = sum(c.isdigit() for c in segment)
    digit_ratio = digits / max(1, len(segment))
    if digit_ratio > 0.3 and len(segment) >= 6:
        return 1
        
    # Check for random-looking combinations
    has_letters = any(c.isalpha() for c in segment)
    has_numbers = any(c.isdigit() for c in segment)
    has_special = any(not c.isalnum() for c in segment)
    if has_letters and has_numbers and has_special and len(segment) >= 8:
        return 1
        
    return 0

def domain_in_subdomains(hostname):
    parts = hostname.split('.')
    # check if registered domain appears twice
    # crude: last two parts as registered domain
    if len(parts) < 3:
        return 0
    reg = '.'.join(parts[-2:])
    return 1 if any(reg in part for part in parts[:-2]) else 0

def https_in_hostname(hostname):
    return 1 if 'https' in hostname.lower() else 0

def detect_brand_names(url):
    """Check for brand names in URL"""
    url_lower = url.lower()
    return sum(1 for brand in BRAND_NAMES if brand in url_lower)

def has_suspicious_tld(hostname):
    """Check if domain uses suspicious TLD"""
    try:
        tld = hostname.split('.')[-1].lower()
        return 1 if tld in SUSPICIOUS_TLDS else 0
    except:
        return 0

def count_sensitive_words(url):
    """Count security/login related words"""
    url_lower = url.lower()
    return sum(1 for word in SENSITIVE_WORDS if word in url_lower)

# ---------------------------
# HTML-based features (needs fetch)
# ---------------------------
def analyze_html_features(url, html_text):
    """Return a dict with approximated features that require page parsing."""
    out = {}
    soup = BeautifulSoup(html_text, "html.parser")
    # Links
    anchors = soup.find_all('a', href=True)
    total_links = len(anchors)
    ext_links = 0
    host = urlparse(url).netloc
    for a in anchors:
        href = a['href']
        parsed = urlparse(href)
        if parsed.netloc and parsed.netloc != "" and parsed.netloc not in host:
            ext_links += 1
    pct_ext_links = (ext_links / total_links) if total_links > 0 else 0.0
    out['PctExtHyperlinks'] = pct_ext_links

    # Resource URLs (img/script/link)
    resources = []
    for tag in soup.find_all(['img', 'script', 'link']):
        href = tag.get('src') or tag.get('href')
        if href:
            resources.append(href)
    total_res = len(resources)
    ext_res = 0
    for r in resources:
        parsed = urlparse(r)
        if parsed.netloc and parsed.netloc != "" and parsed.netloc not in host:
            ext_res += 1
    out['PctExtResourceUrls'] = (ext_res / total_res) if total_res > 0 else 0.0

    # Favicon external?
    favicon = None
    icon_link = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
    if icon_link:
        favicon = icon_link.get('href')
    out['ExtFavicon'] = 0
    if favicon:
        parsed = urlparse(favicon)
        if parsed.netloc and parsed.netloc not in host:
            out['ExtFavicon'] = 1

    # Forms analysis
    forms = soup.find_all('form')
    total_forms = len(forms)
    insecure_forms = 0
    relative_action = 0
    ext_action = 0
    abnormal_action = 0  # action pointing to unusual domains
    for form in forms:
        action = form.get('action')
        if not action:
            relative_action += 1
        else:
            parsed = urlparse(action)
            if parsed.scheme == 'http':
                insecure_forms += 1
            if parsed.netloc and parsed.netloc != "" and parsed.netloc not in host:
                ext_action += 1
            # abnormal: action domain different from host
            if parsed.netloc and parsed.netloc != "" and parsed.netloc != host:
                abnormal_action += 1
    out['InsecureForms'] = 1 if insecure_forms > 0 else 0
    out['RelativeFormAction'] = 1 if relative_action > 0 else 0
    out['ExtFormAction'] = 1 if ext_action > 0 else 0
    out['AbnormalFormAction'] = 1 if abnormal_action > 0 else 0

    # Iframe present?
    out['IframeOrFrame'] = 1 if soup.find(['iframe', 'frame']) else 0

    # Missing title?
    title = soup.title.string if soup.title else None
    out['MissingTitle'] = 1 if not title or title.strip() == "" else 0

    # Images only in form (crude)
    imgs_in_forms = 0
    total_inputs_in_forms = 0
    for f in forms:
        imgs_in_forms += len(f.find_all('img'))
        total_inputs_in_forms += len(f.find_all(['input', 'button', 'textarea', 'select']))
    out['ImagesOnlyInForm'] = 1 if total_inputs_in_forms == 0 and imgs_in_forms > 0 else 0

    # Pct External null/self redirect hyperlinks (approx not exact)
    # we'll set to 0 as approximation
    out['PctExtNullSelfRedirectHyperlinks'] = 0.0

    # Other JS-dependent features we cannot detect reliably
    out['RightClickDisabled'] = 0
    out['PopUpWindow'] = 0
    out['SubmitInfoToEmail'] = 0
    out['FakeLinkInStatusBar'] = 0
    out['FrequentDomainNameMismatch'] = 0

    return out

# ---------------------------
# Build feature vector for a single URL
# ---------------------------
def make_feature_vector_from_url(url, fetch_page=False, timeout=6):
    """Return a dict with features matching your dataset schema.
       For features we cannot compute we set reasonable defaults (0) and mark them in the UI."""
    parsed = urlparse(url if url.startswith(('http://','https://')) else 'http://' + url)
    scheme = parsed.scheme
    host = parsed.netloc
    path = parsed.path
    query = parsed.query

    fv = {}
    fv['id'] = 0
    fv['NumDots'] = count_dots(host)
    fv['SubdomainLevel'] = subdomain_level(host, host)
    fv['PathLevel'] = path_level(path)
    fv['UrlLength'] = url_length(url)
    fv['NumDash'] = count_char(url, '-')
    fv['NumDashInHostname'] = count_char(host, '-')
    fv['AtSymbol'] = has_at_symbol(url)
    fv['TildeSymbol'] = has_tilde(url)
    fv['NumUnderscore'] = count_char(url, '_')
    fv['NumPercent'] = count_char(url, '%')
    fv['NumQueryComponents'] = len(query.split('&')) if query else 0
    fv['NumAmpersand'] = count_char(query, '&')
    fv['NumHash'] = count_char(url, '#')
    fv['NumNumericChars'] = num_chars(url)
    fv['NoHttps'] = 0 if scheme == 'https' else 1
    # RandomString: look for long random path segment
    segments = [seg for seg in path.split('/') if seg]
    fv['RandomString'] = 1 if any(detect_random_string(seg) for seg in segments) else 0
    fv['IpAddress'] = is_ip_address(host)
    fv['DomainInSubdomains'] = domain_in_subdomains(host)
    fv['DomainInPaths'] = 1 if any(host in seg for seg in segments) else 0
    fv['HttpsInHostname'] = https_in_hostname(host)
    fv['HostnameLength'] = hostname_length(host)
    fv['PathLength'] = path_length(path)
    fv['QueryLength'] = len(query)
    fv['DoubleSlashInPath'] = double_slash_in_path(path)
    # Add new features
    fv['NumBrandNames'] = detect_brand_names(url)
    fv['HasSuspiciousTld'] = has_suspicious_tld(host)
    fv['NumSensitiveWords'] = count_sensitive_words(url)
    # Enhance existing features
    fv['SubdomainLevel'] = max(fv['SubdomainLevel'], 3 if fv['SubdomainLevel'] > 2 else fv['SubdomainLevel'])
    fv['RandomString'] = max(fv['RandomString'], 
                           1 if detect_random_string(query) else 0)
    
    # Add length thresholds
    fv['ExcessiveUrlLength'] = 1 if fv['UrlLength'] > 75 else 0
    fv['ExcessiveHostLength'] = 1 if fv['HostnameLength'] > 40 else 0
    
    # For features that require lists of sensitive words / brand names, we set default 0
    fv['EmbeddedBrandName'] = 0
    # HTML-dependent features: initialize defaults
    fv['PctExtHyperlinks'] = 0.0
    fv['PctExtResourceUrls'] = 0.0
    fv['ExtFavicon'] = 0
    fv['InsecureForms'] = 0
    fv['RelativeFormAction'] = 0
    fv['ExtFormAction'] = 0
    fv['AbnormalFormAction'] = 0
    fv['PctNullSelfRedirectHyperlinks'] = 0.0
    fv['FrequentDomainNameMismatch'] = 0
    fv['FakeLinkInStatusBar'] = 0
    fv['RightClickDisabled'] = 0
    fv['PopUpWindow'] = 0
    fv['SubmitInfoToEmail'] = 0
    fv['IframeOrFrame'] = 0
    fv['MissingTitle'] = 0
    fv['ImagesOnlyInForm'] = 0
    fv['SubdomainLevelRT'] = fv['SubdomainLevel']  # runtime approximations
    fv['UrlLengthRT'] = fv['UrlLength']
    fv['PctExtResourceUrlsRT'] = fv['PctExtResourceUrls']
    fv['AbnormalExtFormActionR'] = fv['AbnormalFormAction']
    fv['ExtMetaScriptLinkRT'] = 0
    fv['PctExtNullSelfRedirectHyperlinksRT'] = fv['PctNullSelfRedirectHyperlinks']
    fv['CLASS_LABEL'] = 0  # dummy

    # If page fetch requested, try to compute more features from HTML
    fetch_note = ""
    if fetch_page:
        try:
            r = requests.get(url, timeout=timeout, headers={'User-Agent':'Mozilla/5.0'})
            html = r.text
            html_feats = analyze_html_features(url, html)
            # Map html_feats into fv
            for k,v in html_feats.items():
                if k in fv:
                    fv[k] = v
            # update runtime versions
            fv['PctExtResourceUrlsRT'] = fv.get('PctExtResourceUrls', 0.0)
            fv['AbnormalExtFormActionR'] = fv.get('AbnormalFormAction', 0)
            fv['ExtMetaScriptLinkRT'] = 0  # could be computed by checking <script> src external
            fetch_note = "HTML fetched and parsed."
        except Exception as e:
            fetch_note = f"Failed to fetch page: {e}"
    return fv, fetch_note

# ---------------------------
# Train/load model
# ---------------------------
@st.cache_data(show_spinner=False)
def train_model_from_csv(csv_path, target_col='CLASS_LABEL'):
    df = pd.read_csv(csv_path)
    # expect the dataset has CLASS_LABEL as 1 (phish) / 0 (legit) or similar
    # Keep only columns that the model can accept. We'll use all existing columns except id and CLASS_LABEL for training.
    df_clean = df.dropna(axis=0)  # simple
    X = df_clean.drop(columns=['id', target_col], errors='ignore')
    y = df_clean[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, X.columns.tolist(), acc

# If model file exists, load; else train
model = None
feature_cols = None
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        st.info("Loaded saved model.")
    except Exception:
        model = None

if model is None:
    if not os.path.exists(DATA_CSV):
        st.error(f"Dataset file {DATA_CSV} not found in the app folder. Please add it and reload.")
        st.stop()
    with st.spinner("Training model from CSV (this may take a few seconds)..."):
        model, feature_cols, accuracy = train_model_from_csv(DATA_CSV)
        joblib.dump(model, MODEL_FILE)
        st.success(f"Model trained. Test accuracy ~ {accuracy:.3f}")

# If feature_cols not set by load, try to read from dataset
if feature_cols is None:
    df_tmp = pd.read_csv(DATA_CSV)
    feature_cols = [c for c in df_tmp.columns if c not in ('id','CLASS_LABEL')]

# ---------------------------
# UI: Enter URL and options
# ---------------------------
st.sidebar.header("Options")
fetch_page = st.sidebar.checkbox("Fetch & analyze page HTML (more features)", value=True)
st.sidebar.write("Fetching improves detection but may fail for some sites.")

url_input = st.text_input("Enter URL to check (with or without http/https):", value="http://example.com")
if st.button("Analyze URL"):

    fv, note = make_feature_vector_from_url(url_input.strip(), fetch_page=fetch_page)
    st.write("**Feature extraction note:**", note)
    # Build dataframe with exactly the model's expected columns (order)
    # For missing columns, fill with 0/0.0
    model_input = {}
    for c in feature_cols:
        if c == 'CLASS_LABEL':
            continue
        # if fv has key use it, else try fallback (case differences)
        model_input[c] = fv.get(c, 0)

    X_single = pd.DataFrame([model_input], columns=[c for c in feature_cols if c != 'CLASS_LABEL'])
    st.subheader("Extracted features (used for prediction)")
    st.dataframe(X_single.T)

    # Prediction
    pred_proba = None
    try:
        if hasattr(model, "predict_proba"):
            pred_proba = model.predict_proba(X_single)[0]
            prob_phish = pred_proba[1]
        else:
            prob_phish = float(model.predict(X_single)[0])
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    pred_label = 1 if prob_phish >= 0.5 else 0
    label_text = "🚨 **PHISHING**" if pred_label == 1 else "✅ **LEGITIMATE**"
    st.markdown(f"### Prediction: {label_text}")
    st.write(f"Probability phishing: {prob_phish:.3f}")

    # Explain which features mattered (use feature_importances_)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=[c for c in feature_cols if c != 'CLASS_LABEL'])
        top10 = feat_imp.sort_values(ascending=False).head(10)
        st.subheader("Top features (model importance)")
        st.bar_chart(top10)

    st.info("Note: Some features are approximated. JS-only behaviors (right-click disabled, pop-ups) are not detectable from static HTML and are set to 0.")

st.markdown("---")
st.caption("App built for assignment: feature extraction is best-effort; add more heuristics as needed.")
