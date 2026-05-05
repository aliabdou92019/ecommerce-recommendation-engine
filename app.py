import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Data Mining GUI",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 17px;
    color: #c9d1d9;
    margin-bottom: 30px;
}
.card {
    background-color: #161b22;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #30363d;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.25);
}
.card-title {
    font-size: 15px;
    color: #8b949e;
}
.card-value {
    font-size: 34px;
    font-weight: bold;
    color: #58a6ff;
}
.section-box {
    background-color: #0d1117;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #30363d;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

def load_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

orders = load_csv("Data/orders.csv")
products = load_csv("Data/products.csv")
order_items = load_csv("Data/order_items.csv")
rules = load_csv("Data/Final_rules.csv")
pagerank = load_csv("results/pagerank_scores.csv")
sentiment = load_csv("results/sentiment_results.csv")
recommendations = load_csv("results/final_recommendations.csv")

st.sidebar.title("📊 Data Mining GUI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📁 Data Overview",
        "🔗 Association Rules",
        "🌐 PageRank",
        "💬 Sentiment Analysis",
        "✅ Final Recommendations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Data Mining Project Interface")

if page == "🏠 Home":
    st.markdown('<div class="main-title">Data Mining Project GUI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A professional interface for viewing project datasets, mining results, PageRank scores, sentiment analysis, and final recommendations.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Total Orders</div>
            <div class="card-value">{len(orders)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Total Products</div>
            <div class="card-value">{len(products)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Order Items</div>
            <div class="card-value">{len(order_items)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Project Modules")
    st.markdown("""
    <div class="section-box">
    <b>1. Data Overview:</b> Displays the main datasets used in the project.<br>
    <b>2. Association Rules:</b> Shows generated rules using support and confidence.<br>
    <b>3. PageRank:</b> Displays ranking scores for important nodes/products/pages.<br>
    <b>4. Sentiment Analysis:</b> Shows customer review sentiment results.<br>
    <b>5. Final Recommendations:</b> Presents the final recommendation output.
    </div>
    """, unsafe_allow_html=True)

elif page == "📁 Data Overview":
    st.title("📁 Data Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Orders", len(orders))
    c2.metric("Products", len(products))
    c3.metric("Order Items", len(order_items))

    st.subheader("Orders Data")
    st.dataframe(orders.head(30), use_container_width=True)

    st.subheader("Products Data")
    st.dataframe(products.head(30), use_container_width=True)

    st.subheader("Order Items Data")
    st.dataframe(order_items.head(30), use_container_width=True)

    if not products.empty and "category" in products.columns:
        st.subheader("Products by Category")
        fig = px.histogram(products, x="category")
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔗 Association Rules":
    st.title("🔗 Association Rules")

    if rules.empty:
        st.warning("No association rules found in the file.")
    else:
        st.dataframe(rules, use_container_width=True)

        if "support" in rules.columns and "confidence" in rules.columns:
            rules["support"] = pd.to_numeric(rules["support"], errors="coerce")
            rules["confidence"] = pd.to_numeric(rules["confidence"], errors="coerce")

            clean_rules = rules.dropna(subset=["support", "confidence"])

            if clean_rules.empty:
                st.info("Association rules file exists, but no valid numeric rules are available for plotting.")
            else:
                fig = px.scatter(
                    clean_rules,
                    x="support",
                    y="confidence",
                    title="Support vs Confidence",
                    hover_data=clean_rules.columns
                )
                st.plotly_chart(fig, use_container_width=True)

        if "lift" in rules.columns:
            rules["lift"] = pd.to_numeric(rules["lift"], errors="coerce")
            top_lift = rules.dropna(subset=["lift"]).sort_values("lift", ascending=False).head(10)

            if not top_lift.empty:
                st.subheader("Top Rules by Lift")
                fig = px.bar(top_lift, x=top_lift.index, y="lift")
                st.plotly_chart(fig, use_container_width=True)

elif page == "🌐 PageRank":
    st.title("🌐 PageRank Analysis")

    if pagerank.empty:
        st.warning("No PageRank results found.")
    else:
        st.dataframe(pagerank, use_container_width=True)

        numeric_cols = pagerank.select_dtypes(include="number").columns

        if len(numeric_cols) > 0:
            score_col = numeric_cols[0]
            name_col = pagerank.columns[0]

            top_pages = pagerank.sort_values(score_col, ascending=False).head(10)

            st.subheader("Top PageRank Scores")
            fig = px.bar(
                top_pages,
                x=name_col,
                y=score_col,
                title="Top Ranked Items"
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "💬 Sentiment Analysis":
    st.title("💬 Sentiment Analysis")

    if sentiment.empty:
        st.warning("No sentiment results found.")
    else:
        st.dataframe(sentiment, use_container_width=True)

        if "sentiment" in sentiment.columns:
            sentiment_count = sentiment["sentiment"].value_counts().reset_index()
            sentiment_count.columns = ["Sentiment", "Count"]

            st.subheader("Sentiment Distribution")
            fig = px.pie(
                sentiment_count,
                names="Sentiment",
                values="Count"
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "✅ Final Recommendations":
    st.title("✅ Final Recommendations")
    st.markdown("Enter a product name to get personalized recommendations based on Association Rules and PageRank.")
    
    from recommendation_engine import get_recommendations
    
    product_input = st.text_input("Product Name", "Mouse Chartreuse 292")
    
    if st.button("Get Recommendations"):
        recs_df, message = get_recommendations(product_input, rules, pagerank)
        
        if "Found exact match" in message:
            st.success(message)
        elif "Using closest match" in message:
            st.info(message)
        else:
            st.error(message)
            
        if recs_df is not None and not recs_df.empty:
            st.dataframe(recs_df, use_container_width=True)
            
            # Update the underlying file
            recs_df.to_csv("results/final_recommendations.csv", index=False)
        elif recs_df is not None and recs_df.empty:
            st.warning("No recommendations found for this product.")
            
    st.markdown("---")
    st.subheader("Previous Recommendations Data")
    if not recommendations.empty:
        st.dataframe(recommendations, use_container_width=True)