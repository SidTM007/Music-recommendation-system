"""
Music Recommender - a Linear Algebra project (pure Python)
Run:  pip install streamlit numpy pandas matplotlib
      streamlit run music_recommender.py

Linear algebra used
  vectors / matrices ........ songs are rows of the feature matrix X
  normalisation ............. min-max scaling of every column
  dot product & norms ....... measure alignment and magnitude
  cosine similarity / angle . similarity score between songs
  projection + PCA .......... eigenvectors of the covariance matrix -> 2-D map
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

FEATURES = ["energy", "danceability", "acousticness", "valence", "tempo"]

# Approximate, illustrative values. Replace with a real CSV (sidebar upload).
SONGS = [
    ("Blinding Lights", "The Weeknd", .80, .51, .00, .33, 171), ("Save Your Tears", "The Weeknd", .68, .65, .02, .64, 118),
    ("Levitating", "Dua Lipa", .83, .70, .06, .91, 103), ("Don't Start Now", "Dua Lipa", .79, .79, .01, .68, 124),
    ("Shape of You", "Ed Sheeran", .65, .83, .58, .93, 96), ("Perfect", "Ed Sheeran", .45, .60, .16, .52, 95),
    ("Thinking Out Loud", "Ed Sheeran", .45, .78, .47, .59, 79), ("Watermelon Sugar", "Harry Styles", .82, .55, .12, .56, 95),
    ("Uptown Funk", "Bruno Mars", .61, .86, .01, .93, 115), ("Happy", "Pharrell Williams", .82, .65, .22, .96, 160),
    ("Someone Like You", "Adele", .33, .56, .89, .30, 135), ("Stay With Me", "Sam Smith", .42, .42, .55, .18, 84),
    ("Bad Guy", "Billie Eilish", .43, .70, .33, .56, 135), ("Lovely", "Billie Eilish", .30, .35, .93, .12, 115),
    ("Believer", "Imagine Dragons", .78, .78, .06, .67, 125), ("Radioactive", "Imagine Dragons", .78, .45, .10, .24, 136),
    ("Counting Stars", "OneRepublic", .71, .66, .07, .48, 122), ("Stressed Out", "Twenty One Pilots", .64, .73, .05, .65, 170),
    ("Sunflower", "Post Malone", .48, .76, .56, .93, 90), ("Circles", "Post Malone", .76, .69, .19, .55, 120),
    ("Rockstar", "Post Malone", .52, .59, .12, .12, 160), ("Lose Yourself", "Eminem", .73, .69, .01, .06, 171),
    ("SICKO MODE", "Travis Scott", .73, .83, .01, .45, 155), ("God's Plan", "Drake", .45, .75, .03, .36, 77),
    ("Hotline Bling", "Drake", .62, .89, .01, .81, 135), ("Bohemian Rhapsody", "Queen", .40, .39, .29, .23, 72),
    ("Hotel California", "Eagles", .58, .58, .01, .60, 147), ("Sweet Child O' Mine", "Guns N' Roses", .91, .45, .00, .83, 125),
    ("Smells Like Teen Spirit", "Nirvana", .91, .50, .00, .72, 117), ("Numb", "Linkin Park", .86, .50, .00, .24, 110),
    ("Yesterday", "The Beatles", .18, .34, .88, .32, 97), ("Fix You", "Coldplay", .42, .21, .16, .12, 138),
    ("Viva La Vida", "Coldplay", .62, .49, .09, .42, 138), ("Shallow", "Lady Gaga", .39, .57, .38, .30, 96),
    ("Dance Monkey", "Tones and I", .59, .82, .69, .52, 98), ("Señorita", "Shawn Mendes", .54, .76, .04, .75, 117),
    ("Tum Hi Ho", "Arijit Singh", .45, .52, .60, .30, 86), ("Kala Chashma", "Badshah", .88, .78, .05, .85, 102),
]


# ---------------------------------------------------------------- linear algebra
def min_max(raw):
    """Column-wise min-max normalisation: x' = (x - min) / (max - min)."""
    mn, mx = raw.min(axis=0), raw.max(axis=0)
    return (raw - mn) / (mx - mn), mn, mx


def cosine_all(X, p):
    """cos(theta) between preference vector p and every row of X (one matrix-vector product)."""
    return (X @ p) / (np.linalg.norm(X, axis=1) * np.linalg.norm(p))


def pca(X):
    """Eigen-decomposition of the covariance matrix; returns sorted eigenvalues/vectors."""
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(C)              # symmetric matrix -> eigh
    order = np.argsort(vals)[::-1]
    return Xc, C, vals[order], vecs[:, order]


# ---------------------------------------------------------------- data loading
@st.cache_data
def default_df():
    return pd.DataFrame(SONGS, columns=["name", "artist"] + FEATURES)


st.set_page_config(page_title="Music Recommender", page_icon="🎵", layout="centered")
st.title("🎵 Music Recommender")
st.caption("Songs are vectors. Similar songs point in similar directions.")

with st.sidebar:
    st.header("Dataset")
    up = st.file_uploader("Optional CSV", type="csv",
                          help="Columns needed: name, artist, energy, danceability, acousticness, valence, tempo")
    st.markdown("Using the built-in sample of 38 songs." if up is None else "Using your uploaded CSV.")

df = default_df()
if up is not None:
    try:
        df = pd.read_csv(up)
        df = df[["name", "artist"] + FEATURES].dropna().drop_duplicates("name").reset_index(drop=True)
    except Exception:
        st.sidebar.error("CSV needs columns: name, artist, " + ", ".join(FEATURES))
        df = default_df()

raw = df[FEATURES].to_numpy(dtype=float)
X, col_min, col_max = min_max(raw)
labels = (df["name"] + " — " + df["artist"]).tolist()

# ---------------------------------------------------------------- controls
c1, c2 = st.columns([3, 1])
picked = c1.multiselect("Your favourite song(s)", range(len(df)), default=[0],
                        format_func=lambda i: labels[i], max_selections=5,
                        help="Choose several to build a combined taste profile.")
n_rec = c2.number_input("Recommendations", 1, 15, 5)

if not picked:
    st.info("Pick at least one song to get recommendations.")
    st.stop()

# user preference vector = mean of chosen song vectors
p = X[picked].mean(axis=0)
sims = cosine_all(X, p)
sims[picked] = -np.inf                              # don't recommend what they already chose
top = np.argsort(sims)[::-1][:n_rec]
best = int(top[0])

tab_rec, tab_math, tab_data, tab_pca = st.tabs(
    ["⭐ Recommendations", "🧮 Math walkthrough", "📊 Feature matrix", "🗺️ PCA map"])

# ---------------------------------------------------------------- tab 1
with tab_rec:
    st.subheader("Recommended for you")
    for rank, i in enumerate(top, 1):
        a, b, c = st.columns([0.6, 5, 1.2])
        a.markdown(f"### {rank}")
        b.markdown(f"**{df['name'][i]}**  \n{df['artist'][i]}")
        b.progress(float(sims[i]))
        c.markdown(f"### {sims[i]:.2f}")
    st.caption("Score = cosine similarity (1.00 = same direction). "
               "See the Math walkthrough tab for how the #1 result was computed.")

# ---------------------------------------------------------------- tab 2
with tab_math:
    a_vec, b_vec = p, X[best]
    dp = float(a_vec @ b_vec)
    na, nb = float(np.linalg.norm(a_vec)), float(np.linalg.norm(b_vec))
    cos = dp / (na * nb)
    ang = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    st.subheader(f"Why “{df['name'][best]}” is #1")
    fmt = lambda v: "[" + ", ".join(f"{x:.3f}" for x in v) + "]"

    st.markdown("**Step 1 — Vector representation.** Each song is a point in 5-D space "
                f"`[{', '.join(FEATURES)}]`.")
    st.code(f"Recommended song (raw) = {raw[best].tolist()}")

    st.markdown("**Step 2 — Normalisation.** Tempo (BPM) is much larger than the 0–1 features and would "
                "dominate the dot product, so each column is rescaled:")
    st.latex(r"x' = \frac{x-\min}{\max-\min}")
    st.code(f"Recommended song (normalised) = {fmt(b_vec)}")

    st.markdown("**Step 3 — User preference vector.** The average of your chosen song vectors.")
    st.code(f"P = {fmt(a_vec)}")

    st.markdown("**Step 4 — Dot product (feature alignment).**")
    st.code(" + ".join(f"{x:.3f}×{y:.3f}" for x, y in zip(a_vec, b_vec)) + f"\n= {dp:.4f}")

    st.markdown("**Step 5 — Norms (vector magnitude).**")
    st.latex(r"\|v\|=\sqrt{v\cdot v}")
    st.code(f"‖P‖ = {na:.4f}     ‖S‖ = {nb:.4f}")

    st.markdown("**Step 6 — Cosine similarity and angle.**")
    st.latex(r"\cos\theta=\frac{P\cdot S}{\|P\|\,\|S\|}")
    st.success(f"cos θ = {dp:.4f} / ({na:.4f} × {nb:.4f}) = {cos:.4f}   →   θ = {ang:.2f}°")

    st.markdown("**Step 7 — Ranking.** All songs are scored at once with one matrix–vector product "
                "`X′·P` (divided by the norms), then sorted in descending order.")

# ---------------------------------------------------------------- tab 3
with tab_data:
    st.subheader(f"Raw feature matrix ({raw.shape[0]}×{raw.shape[1]})")
    st.dataframe(df.set_index("name")[FEATURES], width='stretch')
    st.subheader("Normalised matrix X′")
    st.caption(f"Column min = {np.round(col_min, 2).tolist()}   |   column max = {np.round(col_max, 2).tolist()}")
    st.dataframe(pd.DataFrame(X, index=df["name"], columns=FEATURES).round(3), width='stretch')

# ---------------------------------------------------------------- tab 4
with tab_pca:
    Xc, C, vals, vecs = pca(X)
    P2 = Xc @ vecs[:, :2]                           # projection onto the top-2 eigenvectors
    explained = vals / vals.sum()
    st.subheader("2-D map of the music space")
    st.caption("PCA projects the 5 features onto the 2 directions of greatest variance "
               "(the top eigenvectors of the covariance matrix).")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(P2[:, 0], P2[:, 1], c="#bbbbbb", s=25)
    ax.scatter(P2[top, 0], P2[top, 1], c="#16a34a", s=70, label="Recommended")
    ax.scatter(P2[picked, 0], P2[picked, 1], c="#ef4444", s=90, label="Your choice")
    for i in list(top) + picked:
        ax.annotate(df["name"][i], P2[i], xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(f"PC1 ({explained[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} variance)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    st.markdown("**Covariance matrix C = X′ᶜᵀ X′ᶜ / (n−1)**")
    st.dataframe(pd.DataFrame(C, index=FEATURES, columns=FEATURES).round(4))
    st.markdown("**Eigenvalues and explained variance**")
    st.dataframe(pd.DataFrame({"eigenvalue": vals.round(4), "variance %": (explained * 100).round(1)},
                              index=[f"PC{k+1}" for k in range(len(vals))]))
    st.markdown("**Top-2 eigenvectors (loadings)**")
    st.dataframe(pd.DataFrame(vecs[:, :2], index=FEATURES, columns=["PC1", "PC2"]).round(3))
    st.success(f"Top 2 components keep {explained[:2].sum():.1%} of the information.")
