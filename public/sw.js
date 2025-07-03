const CACHE_NAME = 'yousef-sh-agent-v1';
const OFFLINE_URL = '/offline.html';

// Files to cache for offline functionality
const CACHE_FILES = [
  '/',
  '/styles.css',
  '/script.js',
  '/offline.html',
  'https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap'
];

// Install event - cache essential files
self.addEventListener('install', (event) => {
  console.log('[SW] Install Event processing');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] Caching app shell');
        return cache.addAll(CACHE_FILES);
      })
      .catch((error) => {
        console.error('[SW] Cache installation failed:', error);
      })
  );
  
  // Force the waiting service worker to become the active service worker
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activate Event processing');
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (cacheName !== CACHE_NAME) {
              console.log('[SW] Removing old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        // Take control of all clients immediately
        return self.clients.claim();
      })
  );
});

// Fetch event - serve cached content when offline
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }
  
  // Skip Chrome extension requests
  if (url.protocol === 'chrome-extension:') {
    return;
  }
  
  // Handle API requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(request)
        .catch(() => {
          // Return offline response for API calls
          return new Response(
            JSON.stringify({
              error: 'أنت غير متصل بالإنترنت حالياً',
              offline: true
            }),
            {
              status: 503,
              statusText: 'Service Unavailable',
              headers: {
                'Content-Type': 'application/json'
              }
            }
          );
        })
    );
    return;
  }
  
  // Handle Socket.io requests
  if (url.pathname.startsWith('/socket.io/')) {
    event.respondWith(
      fetch(request)
        .catch(() => {
          // Return empty response for socket.io when offline
          return new Response('', { status: 404 });
        })
    );
    return;
  }
  
  // Handle page requests
  event.respondWith(
    caches.match(request)
      .then((cachedResponse) => {
        // Return cached version if available
        if (cachedResponse) {
          return cachedResponse;
        }
        
        // Try to fetch from network
        return fetch(request)
          .then((response) => {
            // Check if we received a valid response
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }
            
            // Clone the response for caching
            const responseToCache = response.clone();
            
            // Cache the new response
            caches.open(CACHE_NAME)
              .then((cache) => {
                cache.put(request, responseToCache);
              });
            
            return response;
          })
          .catch(() => {
            // Return offline page for navigation requests
            if (request.mode === 'navigate') {
              return caches.match(OFFLINE_URL);
            }
            
            // Return a generic offline response for other requests
            return new Response('أنت غير متصل بالإنترنت', {
              status: 503,
              statusText: 'Service Unavailable',
              headers: {
                'Content-Type': 'text/plain; charset=utf-8'
              }
            });
          });
      })
  );
});

// Handle background sync (if supported)
self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync:', event.tag);
  
  if (event.tag === 'background-sync') {
    event.waitUntil(
      // Perform background sync operations
      syncData()
    );
  }
});

// Handle push notifications (if supported)
self.addEventListener('push', (event) => {
  console.log('[SW] Push notification received');
  
  const options = {
    body: event.data ? event.data.text() : 'رسالة جديدة من YOUSEF SH',
    icon: '/icon-192x192.png',
    badge: '/badge-72x72.png',
    tag: 'yousef-sh-notification',
    requireInteraction: true,
    actions: [
      {
        action: 'open',
        title: 'فتح التطبيق'
      },
      {
        action: 'close',
        title: 'إغلاق'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification('YOUSEF SH', options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  console.log('[SW] Notification click received');
  
  event.notification.close();
  
  if (event.action === 'open') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Background sync function
async function syncData() {
  try {
    // Implement background sync logic here
    console.log('[SW] Performing background sync');
    
    // Example: sync pending messages, check for updates, etc.
    
  } catch (error) {
    console.error('[SW] Background sync failed:', error);
  }
}

// Message handling from the main thread
self.addEventListener('message', (event) => {
  console.log('[SW] Message received:', event.data);
  
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data && event.data.type === 'GET_VERSION') {
    event.ports[0].postMessage({ version: CACHE_NAME });
  }
});