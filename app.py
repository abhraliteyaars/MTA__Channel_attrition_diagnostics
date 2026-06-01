import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from snowflake.snowpark.context import get_active_session

st.set_page_config(
    page_title="LNDC → Markov Revenue Reallocation Diagnostics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

session = get_active_session()
query = "SELECT * FROM ECOMM_ANALYTICS.GA360_DEV.MTA_LNDC_ORDER_LEVEL_DATA_VW"
df_main = session.sql(query).to_pandas()


def normalize_alloc_columns(alloc: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in alloc.columns:
        if col.startswith("MO_"):
            rename_map[col] = "MO"
        elif col.startswith("TM_"):
            rename_map[col] = "TM"

    if rename_map:
        alloc.columns = [rename_map.get(col, col) for col in alloc.columns]
        alloc = alloc.T.groupby(level=0).sum().T

    return alloc


def filter_data(country, tx_devices, date_range, ftb_values):
    df = df_main.copy()

    df = df[df["COUNTRYCODE"] == country]
    df = df[df["TX_DEVICE"].isin(tx_devices)]
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"])

    start_date, end_date = date_range
    df = df[
        (df["ORDERDATE"] >= pd.Timestamp(start_date))
        & (df["ORDERDATE"] <= pd.Timestamp(end_date))
    ]

    df = df[df["FTBIND"].isin(ftb_values)]

    email_map = {"EMAIL_TX": "Email Triggers", "Email Triggers": "Email Triggers"}
    df["CHANNELS"] = df["CHANNELS"].replace(email_map)
    df["LNDC_CHANNEL"] = df["LNDC_CHANNEL"].replace(email_map)

    grain_cols = [
        "ORDERDATE", "FTBIND", "COUNTRYCODE", "TX_DEVICE",
        "SOURCE", "MEDIUM", "CAMPAIGN", "CHANNELS", "SALESCHANNEL",
    ]

    dd_agg = (
        df.groupby(grain_cols + ["LNDC_CHANNEL"])["DD_MARGINAL_REVENUE"]
        .sum()
        .reset_index()
    )

    alloc = (
        dd_agg.groupby(["LNDC_CHANNEL", "CHANNELS"])["DD_MARGINAL_REVENUE"]
        .sum()
        .unstack(fill_value=0)
    )
    alloc = normalize_alloc_columns(alloc)

    lndc_agg = (
        df.groupby(grain_cols + ["LNDC_CHANNEL"])["MARGINAL_REVENUE"]
        .sum()
        .reset_index()
    )

    totals = (
        lndc_agg.groupby("LNDC_CHANNEL")["MARGINAL_REVENUE"]
        .sum()
        .to_frame("LNDC_REV")
    )

    return alloc, totals


def build_counterparty_waterfall(lndc_label, net_df, old_rev, final_rev):
    if net_df.empty:
        return None

    plot_df = net_df.copy()
    plot_df = plot_df.sort_values("Net_Flow", ascending=True)

    labels = ["Old LNDC Revenue"] + plot_df["Counterparty"].tolist() + ["Final Markov Revenue"]
    measures = ["absolute"] + ["relative"] * len(plot_df) + ["total"]

    y_vals = [old_rev] + plot_df["Net_Flow"].tolist() + [0]

    text_vals = [f"${old_rev:,.0f}"]
    for v in plot_df["Net_Flow"].tolist():
        if v >= 0:
            text_vals.append(f"+${v:,.0f}")
        else:
            text_vals.append(f"-${abs(v):,.0f}")
    text_vals.append(f"${final_rev:,.0f}")

    fig = go.Figure(
        go.Waterfall(
            name="Net Counterparty Flow",
            orientation="v",
            measure=measures,
            x=labels,
            y=y_vals,
            text=text_vals,
            textposition="outside",
            connector={"line": {"color": "rgba(90, 90, 90, 0.35)"}},
            increasing={"marker": {"color": "#2ca02c"}},
            decreasing={"marker": {"color": "#d62728"}},
            totals={"marker": {"color": "#1f77b4"}},
        )
    )

    fig.update_layout(
        title=f"Net Counterpart Waterfall for {lndc_label}",
        height=max(500, 40 * (len(plot_df) + 2)),
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )

    return fig


def main():
    st.title("LNDC → Markov Revenue Reallocation")
    st.markdown(
        "Select an original LNDC channel to see how its revenue is reallocated under the Markov model."
    )

    # st.markdown("### Controls")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        countries = sorted(df_main['COUNTRYCODE'].unique())
        country = st.selectbox("Country", countries, index=0)

    with c2:
        all_devices = sorted(df_main['TX_DEVICE'].unique())
        selected_devices = st.multiselect("Device", all_devices)
        if not selected_devices:
            selected_devices = all_devices

    with c3:
        date_range = st.date_input(
            "Date range",
            value=(
                pd.Timestamp("2026-01-01").date(),
                pd.Timestamp("2026-01-31").date(),
            ),
        )

    with c4:
        ftb_labels = ["All", "FTB=0", "FTB=1"]
        ftb_selected = st.selectbox("FTB flag", ftb_labels, index=0)

        if ftb_selected == "All":
            ftb_values = [0, 1]
        elif ftb_selected == "FTB=0":
            ftb_values = [0]
        else:
            ftb_values = [1]

    alloc, totals = filter_data(country, selected_devices, date_range, ftb_values)

    lndc_options = list(totals.index) if len(totals) > 0 else []
    if not lndc_options:
        st.error("No data available with the selected filters.")
        return

    lndc = st.selectbox("LNDC channel (detail)", lndc_options, index=0)

    st.header("Stakeholder grid: LNDC → Contributing channels")

    all_lndc = sorted(totals.index)
    all_channels = sorted(alloc.columns)

    grid = alloc.reindex(index=all_lndc, columns=all_channels).fillna(0).copy()
    grid["LNDC Rev"] = [
        float(totals.loc[idx, "LNDC_REV"]) if idx in totals.index else grid.loc[idx].sum()
        for idx in grid.index
    ]

    percent_grid = grid.drop(columns=["LNDC Rev"]).div(grid["LNDC Rev"], axis=0).fillna(0)
    percent_grid["Row total"] = grid["LNDC Rev"]

    total_rev = grid["LNDC Rev"].sum()
    dd_rev = grid.drop(columns=["LNDC Rev"]).sum(axis=0)
    bottom_row = dd_rev.copy()
    bottom_row["Row total"] = total_rev

    display_grid = percent_grid.copy()
    for col in all_channels:
        display_grid[col] = display_grid[col].map(lambda x: f"{x:.1%}")
    display_grid["Row total"] = display_grid["Row total"].map(lambda x: f"{x:,.0f}")

    bottom_display = bottom_row.map(lambda x: f"{x:,.0f}")
    display_grid = pd.concat(
        [display_grid, pd.DataFrame(bottom_display).T.rename(index={0: "DD Rev"})]
    )

    col1, col2, col3 = st.columns(3)

    if lndc in totals.index:
        lndc_before = float(totals.loc[lndc, "LNDC_REV"])
        col_total = float(dd_rev.get(lndc, 0.0))
        net_change = col_total - lndc_before
        pct_change = (net_change / lndc_before * 100) if lndc_before != 0 else 0

        with col1:
            st.metric("LNDC Revenue (before)", f"${lndc_before:,.0f}")

        with col2:
            st.metric("DD Revenue (after)", f"${col_total:,.0f}")

        with col3:
            color = "green" if net_change >= 0 else "red"
            st.markdown(
                f"""
                <div style="padding: 0.4rem 0;">
                    <div style="font-size: 0.9rem; color: #666;">Net Change</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: {color};">
                        ${net_change:,.0f} ({pct_change:+.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write(
        "This grid shows component allocations as percentages, while row totals and the bottom row are shown as raw dollar totals."
    )
    st.dataframe(display_grid, use_container_width=True)

    if lndc in totals.index:
        st.subheader(f"Net counterpart waterfall for {lndc}")

        if lndc in grid.index and lndc in grid.columns:
            outflow_row = grid.loc[lndc].drop("LNDC Rev", errors="ignore")
            inflow_col = grid[lndc].drop(lndc, errors="ignore")

            counterparties = sorted(set(outflow_row.index).union(set(inflow_col.index)))

            rows = []
            for ch in counterparties:
                selected_to_counterparty = float(outflow_row.get(ch, 0.0))
                counterparty_to_selected = float(inflow_col.get(ch, 0.0))
                net_flow = counterparty_to_selected - selected_to_counterparty

                if (
                    selected_to_counterparty != 0
                    or counterparty_to_selected != 0
                    or net_flow != 0
                ):
                    rows.append(
                        {
                            "Counterparty": ch,
                            "Selected_to_Counterparty": selected_to_counterparty,
                            "Counterparty_to_Selected": counterparty_to_selected,
                            "Net_Flow": net_flow,
                        }
                    )

            net_df = pd.DataFrame(rows)

            if not net_df.empty:
                net_df = net_df.sort_values("Net_Flow", ascending=True)

                net_waterfall = build_counterparty_waterfall(
                    lndc_label=lndc,
                    net_df=net_df,
                    old_rev=float(totals.loc[lndc, "LNDC_REV"]),
                    final_rev=float(grid[lndc].sum()) if lndc in grid.columns else 0.0,
                )

                if net_waterfall is not None:
                    st.plotly_chart(net_waterfall, use_container_width=True)
                else:
                    st.info("No net counterpart flow found for the selected channel.")
            else:
                st.info("No net counterpart flow found for the selected channel.")
        else:
            st.info("Selected channel is not present in both rows and columns of the matrix.")


if __name__ == "__main__":
    main()
