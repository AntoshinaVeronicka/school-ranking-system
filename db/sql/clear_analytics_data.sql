BEGIN;

TRUNCATE TABLE
    edu.analytics_school_report,
    edu.analytics_school_card,
    edu.analytics_school_metric_value,
    edu.analytics_school_rating,
    edu.analytics_run_school,
    edu.analytics_run_metric,
    edu.analytics_school_selection,
    edu.analytics_request_filter,
    edu.analytics_rating_run,
    edu.analytics_request
RESTART IDENTITY;

COMMIT;
