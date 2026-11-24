# Data browser

Read-only Streamlit browser for `data/bronze`, `data/silver` and `data/gold`.
Caddy sits in front of it and asks for a login. Streamlit itself publishes no port.

    # set or change a login (asks for the password)
    tools/dashboard/set_user.sh <username>

    # update after code changes
    docker compose up -d --build --no-deps dashboard

    # address
    https://<DASHBOARD_HOST>  (set in caddy/private/host.env, not in git)

    # rollback
    docker compose rm -sf caddy dashboard

Remove a user by deleting their line in `caddy/private/users.caddy` (not in git), then
`docker compose up -d --no-deps --force-recreate caddy`.

The certificate is a 6-day Let's Encrypt IP certificate that Caddy renews by itself, so ports
80 and 443 must stay reachable. If the IP or hostname changes, update `caddy/private/host.env`.

Tests: `pytest dashboard/tests`
