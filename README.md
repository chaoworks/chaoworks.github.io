# ChaoWorks

Source for [chaoworks.github.io](https://chaoworks.github.io/), a personal engineering blog about LLM systems, inference, and applied AI.

## Website analytics

Google Analytics 4 (GA4) is supported through `_includes/analytics.html`.
Collection remains disabled until `google_analytics` in `_config.yml` contains
the real web stream measurement ID (`G-...`). The measurement ID is public;
account passwords, API secrets, and credentials must not be added to this repo.

### Activate

1. In [Google Analytics](https://analytics.google.com/), create or select a GA4
   property for ChaoWorks. Use Singapore as the reporting time zone.
2. Create a **Web** data stream for `https://chaoworks.github.io` and copy its
   **Measurement ID**, available under **Admin → Data streams → ChaoWorks**.
3. Set `google_analytics: "G-YOUR-ACTUAL-ID"` in `_config.yml` and deploy to
   GitHub Pages. Do not deploy the example ID above.
4. Open an article and verify the visit in the property's **Realtime** report.
   This verification visit is real traffic and should be noted when recording
   the initial baseline. Standard reports may take time to populate.

The shared layout tags English and Chinese reading pages. The automatic
language redirect at `/` is not counted separately, avoiding two page views
for one arrival. Its query parameters and fragment are preserved so campaign
links (`utm_source`, etc.) survive the redirect. Advertising personalization
and Google signals are disabled. Local development builds do not load GA4;
the production build uses `JEKYLL_ENV=production`, as GitHub Pages does.

### Read and preserve reports

Use **Pages and screens** for article views, **Traffic acquisition** for visit
sources, and the report's user metrics for estimated visitors. Views and users
are different metrics; user counts do not identify individual people and can
be affected by browser settings or blockers.

For a monthly record, export the relevant reports as CSV or PDF from the
report's sharing/export menu. Keep the original export together with the
property name, exact date range, reporting time zone, metric definitions, and
any active filters. Do not sum monthly user counts and label that total as
unique users across the whole period. Keep exported reports outside this public
repository. There is no historical website traffic to backfill before activation.

Official guides: [set up GA4](https://support.google.com/analytics/answer/9304153),
[find the measurement ID](https://support.google.com/analytics/answer/9539598),
[export reports](https://support.google.com/analytics/answer/9317657).
