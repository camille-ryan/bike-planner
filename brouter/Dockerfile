FROM eclipse-temurin:17-jre-alpine

ARG BROUTER_VERSION=1.7.9
ENV BROUTER_DIR=/opt/brouter

WORKDIR ${BROUTER_DIR}

# Pull the published BRouter release. The zip extracts to brouter-${VER}/
# containing brouter-${VER}-all.jar, lib/, and profiles2/ (default profiles).
RUN apk add --no-cache --virtual .build-deps curl unzip \
 && curl -fsSL "https://github.com/abrensch/brouter/releases/download/v${BROUTER_VERSION}/brouter-${BROUTER_VERSION}.zip" -o /tmp/brouter.zip \
 && unzip -q /tmp/brouter.zip -d /tmp/brouter \
 && mv /tmp/brouter/brouter-${BROUTER_VERSION}/* . \
 && mv brouter-${BROUTER_VERSION}-all.jar brouter.jar \
 && rm -f BRouterApp*.apk \
 && rm -rf /tmp/brouter /tmp/brouter.zip \
 && apk del .build-deps \
 && mkdir -p customprofiles segments4

EXPOSE 17777

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# RouteServer args: <segments-dir> <profiles-dir> <customprofiles-dir> <port> <max-threads>
# Heap + maxRunningTime are bumped well past the upstream defaults (512M / 300s)
# because the corridor here can plan multi-hundred-km legs across 9 segment
# tiles. Long routes (>~600 km point-to-point) should still use intermediate
# via-points in the URL — BRouter's search is exponential in distance.
CMD ["java", "-Xmx4G", "-Xms256M", "-DmaxRunningTime=1800", \
     "-cp", "brouter.jar:lib/*", "btools.server.RouteServer", \
     "segments4", "profiles2", "customprofiles", "17777", "4"]
