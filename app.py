import streamlit as st

st.set_page_config(page_title="GENIE - Create Dataset", page_icon="🌟")

def main():
    # App title and catchphrase
    st.title("🌟 GENIE 🌟")
    st.subheader("GENerative Information Engine")
    st.markdown("**Generate synthetic data in seconds - no strings attached!**")

    # Instructions
    st.markdown("### How to Use GENIE:")
    st.markdown("1. **Choose a title** for your dataset. 📝")
    st.markdown("2. **Add your preferred fields** one by one. ➕")
    st.markdown("3. **Click the 'Generate Dataset' button** to create your synthetic dataset. 🚀")
    st.markdown("4. The more the number of rows and fields, the more time it will take, so a little patience will be appreciated. 🕒")

    # Add some space
    st.markdown("---")
    

    # Create dataset button
    if st.button("🎯 Create Dataset", use_container_width=True):
        st.switch_page("pages/1_create.py")

if __name__ == "__main__":
    main()
