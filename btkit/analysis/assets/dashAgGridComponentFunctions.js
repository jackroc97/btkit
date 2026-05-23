var dagcomponentfuncs = window.dashAgGridComponentFunctions =
    window.dashAgGridComponentFunctions || {};

dagcomponentfuncs.ChartLink = function (props) {
    if (!props.value) return null;
    return React.createElement(
        "a",
        {
            href: "/chart/" + props.value,
            target: "_blank",
            rel: "noopener noreferrer",
            title: "Open chart",
            style: {
                display: "block",
                textAlign: "center",
                textDecoration: "none",
                fontSize: "15px",
                lineHeight: "32px",
            },
        },
        "📈"
    );
};
